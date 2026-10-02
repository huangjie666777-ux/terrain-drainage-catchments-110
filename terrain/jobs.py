"""Persistent background earthwork jobs with cancel and restart handling.

Job records live in <data_dir>/earthwork/jobs.json; completed results live
in <data_dir>/earthwork/<job_id>/. Workers write into a temporary directory
and atomically promote it on success, so failed or cancelled jobs never
publish partial results. On startup, jobs left in queued/running state are
marked 'interrupted'; completed results stay downloadable.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from affine import Affine

from .config import settings
from .earthwork import (
    GridSpec,
    JobCancelled,
    aligned_grid,
    run_earthwork,
    validate_region,
    validate_utm_epsg,
)
from .storage import DatasetMeta, Registry

ACTIVE_STATES = ("queued", "running")
TERMINAL_STATES = ("completed", "failed", "cancelled", "interrupted")


def _meta_from_record(record: dict[str, Any]) -> DatasetMeta:
    record = dict(record)
    record["bounds"] = tuple(record["bounds"])
    return DatasetMeta(**record)


class JobManager:
    def __init__(
        self,
        registry: Registry,
        root: Path | str | None = None,
        max_workers: int | None = None,
    ) -> None:
        self.registry = registry
        self.root = Path(root) if root is not None else settings.data_dir / "earthwork"
        self.root.mkdir(parents=True, exist_ok=True)
        self.jobs_path = self.root / "jobs.json"
        self._lock = threading.RLock()
        self._cancel: dict[str, threading.Event] = {}
        self._jobs: dict[str, dict[str, Any]] = self._load()
        changed = False
        for job in self._jobs.values():
            if job["status"] in ACTIVE_STATES:
                job["status"] = "interrupted"
                job["error"] = "service restarted before the job finished"
                changed = True
                self._cleanup(job["id"])
        if changed:
            self._save()
        workers = max_workers if max_workers is not None else settings.max_concurrent_jobs
        self._pool = ThreadPoolExecutor(
            max_workers=max(1, workers), thread_name_prefix="earthwork"
        )

    def _load(self) -> dict[str, dict[str, Any]]:
        try:
            with self.jobs_path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except FileNotFoundError:
            return {}

    def _save(self) -> None:
        tmp = self.jobs_path.with_name(f".jobs.{uuid.uuid4().hex}.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(self._jobs, fh, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.jobs_path)

    def _job_dir(self, job_id: str) -> Path:
        return self.root / job_id

    def _tmp_dir(self, job_id: str) -> Path:
        return self.root / f".tmp-{job_id}"

    def _cleanup(self, job_id: str) -> None:
        shutil.rmtree(self._tmp_dir(job_id), ignore_errors=True)
        job = self._jobs.get(job_id)
        if job is None or job["status"] != "completed":
            shutil.rmtree(self._job_dir(job_id), ignore_errors=True)

    def submit(
        self,
        before_version: str,
        after_version: str,
        region_geojson: Any,
        epsg: int,
        resolution: float,
    ) -> dict[str, Any]:
        """Validate the request, persist a queued job, and schedule it.

        Raises VersionError for unknown versions and EarthworkError for
        invalid region/CRS/resolution/grid-size inputs.
        """
        before = self.registry.version(before_version)
        after = self.registry.version(after_version)
        validate_utm_epsg(epsg)
        region = validate_region(region_geojson)
        grid = aligned_grid(region, resolution, int(epsg))
        job_id = uuid.uuid4().hex
        record = {
            "id": job_id,
            "status": "queued",
            "error": None,
            "created_at": time.time(),
            "updated_at": time.time(),
            "before_version": before_version,
            "after_version": after_version,
            "before_datasets": [meta.to_dict() | {"id": meta.id} for meta in before],
            "after_datasets": [meta.to_dict() | {"id": meta.id} for meta in after],
            "region": region_geojson,
            "epsg": int(epsg),
            "resolution": float(resolution),
            "grid": grid.to_dict(),
            "progress": {"rows_done": 0, "rows_total": grid.height},
            "result": None,
        }
        with self._lock:
            self._jobs[job_id] = record
            self._save()
            self._cancel[job_id] = threading.Event()
        self._pool.submit(self._run, job_id)
        return record

    def _update(self, job_id: str, **fields: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.update(fields)
            job["updated_at"] = time.time()
            self._save()

    def _run(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job["status"] != "queued":
                return
            job["status"] = "running"
            job["updated_at"] = time.time()
            self._save()
            cancel_event = self._cancel[job_id]
            before = [_meta_from_record(r) for r in job["before_datasets"]]
            after = [_meta_from_record(r) for r in job["after_datasets"]]
            region = validate_region(job["region"])
            grid_info = job["grid"]
        left, _bottom, _right, top = grid_info["bounds"]
        grid = GridSpec(
            epsg=grid_info["epsg"],
            resolution=grid_info["resolution"],
            transform=Affine(
                grid_info["resolution"], 0.0, left,
                0.0, -grid_info["resolution"], top,
            ),
            width=grid_info["width"],
            height=grid_info["height"],
        )
        tmp_dir = self._tmp_dir(job_id)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_diff = tmp_dir / "diff.tif"

        def on_progress(rows_done: int, rows_total: int) -> None:
            with self._lock:
                current = self._jobs.get(job_id)
                if current is not None:
                    current["progress"] = {
                        "rows_done": rows_done,
                        "rows_total": rows_total,
                    }

        try:
            summary = run_earthwork(
                before,
                after,
                grid,
                region,
                str(self.registry.dataset_dir),
                tmp_diff,
                cancel_event=cancel_event,
                on_progress=on_progress,
            )
            final_dir = self._job_dir(job_id)
            final_dir.mkdir(parents=True, exist_ok=True)
            os.replace(tmp_diff, final_dir / "diff.tif")
            with (final_dir / "result.json").open("w", encoding="utf-8") as fh:
                json.dump(summary, fh, indent=2, sort_keys=True)
            shutil.rmtree(tmp_dir, ignore_errors=True)
            self._update(job_id, status="completed", result=summary, error=None)
        except JobCancelled:
            self._cleanup(job_id)
            self._update(job_id, status="cancelled", error="cancelled by user")
        except Exception as exc:  # persisted for the client
            self._cleanup(job_id)
            self._update(job_id, status="failed", error=str(exc))
        finally:
            with self._lock:
                self._cancel.pop(job_id, None)

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job is not None else None

    def cancel(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if job["status"] in ACTIVE_STATES:
                event = self._cancel.get(job_id)
                if event is not None:
                    event.set()
            return dict(job)

    def diff_path(self, job_id: str) -> Path | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job["status"] != "completed":
                return None
        path = self._job_dir(job_id) / "diff.tif"
        return path if path.exists() else None

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=False)
