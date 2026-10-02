"""Persistent dataset and immutable published-version registry."""

from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.errors import CRSError, RasterioIOError

from .config import settings

ALLOWED_EPSG = (4326, 3857)


def is_utm_crs(crs: CRS | None) -> bool:
    """True for projected UTM metre CRS (e.g. EPSG:32647)."""
    if crs is None or not crs.is_projected:
        return False
    unit, _factor = crs.units_factor
    if unit != "metre":
        return False
    return "UTM" in crs.to_wkt().upper()


class UploadError(ValueError):
    """Raised when an uploaded raster cannot be accepted."""


class VersionError(ValueError):
    """Raised for invalid or conflicting publication requests."""


@dataclass(frozen=True)
class DatasetMeta:
    id: str
    filename: str
    bytes: int
    width: int
    height: int
    bands: int
    crs: str
    dtype: str
    nodata: float | None
    bounds: tuple[float, float, float, float]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["bounds"] = list(self.bounds)
        return value


def _epsg(crs: CRS | None) -> int | None:
    if crs is None or not crs.is_epsg_code:
        return None
    try:
        return int(crs.to_epsg())
    except (CRSError, ValueError, TypeError):
        return None


def validate_raster(path: Path) -> DatasetMeta:
    """Open and validate a candidate single-band raster."""
    try:
        with rasterio.open(path) as src:
            crs = src.crs
            code = _epsg(crs)
            if code not in ALLOWED_EPSG and not is_utm_crs(crs):
                raise UploadError(
                    "only EPSG:4326, EPSG:3857, or UTM projected rasters are accepted"
                )
            if src.count != 1:
                raise UploadError(f"exactly one band required, got {src.count}")
            transform = src.transform
            if transform is None or transform == Affine.identity() or transform.determinant == 0:
                raise UploadError("raster has no valid affine transform")
            width, height = src.width, src.height
            if width < 2 or height < 2:
                raise UploadError("raster is too small")
            # Touch every compressed block without holding the full raster in
            # memory. Tile rendering separately reads only the tile window.
            checked_samples = 0
            for _block_index, window in src.block_windows(1):
                checked_samples += src.read(1, window=window, masked=True).count()
            if checked_samples == 0:
                raise UploadError("raster contains no samples")
            nodata = src.nodata
            dtype = src.dtypes[0]
            bounds = tuple(src.bounds)
    except (RasterioIOError, OSError, ValueError) as exc:
        if isinstance(exc, UploadError):
            raise
        raise UploadError(f"corrupt or unreadable GeoTIFF: {exc}") from exc

    return DatasetMeta(
        id="",
        filename=path.name,
        bytes=path.stat().st_size,
        width=width,
        height=height,
        bands=1,
        crs=f"EPSG:{code}",
        dtype=dtype,
        nodata=None if nodata is None else float(nodata),
        bounds=bounds,
    )


class Registry:
    """File-backed registry safe for concurrent readers and one writer."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or settings.data_dir
        self.dataset_dir = self.root / "datasets"
        self.tmp_dir = self.root / "tmp"
        self.registry_path = self.root / "registry.json"
        self._lock = threading.RLock()
        self.root.mkdir(parents=True, exist_ok=True)
        self.dataset_dir.mkdir(exist_ok=True)
        self.tmp_dir.mkdir(exist_ok=True)

    def _load(self) -> dict[str, Any]:
        try:
            with self.registry_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except FileNotFoundError:
            return {"datasets": {}, "versions": {}}
        data.setdefault("datasets", {})
        data.setdefault("versions", {})
        return data

    def _atomic_json(self, data: dict[str, Any]) -> None:
        tmp = self.registry_path.with_name(f".registry.{uuid.uuid4().hex}.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.registry_path)

    def dataset_path(self, dataset_id: str) -> Path:
        return self.dataset_dir / f"{dataset_id}.tif"

    def ingest(self, filename: str, payload: bytes) -> DatasetMeta:
        if len(payload) > settings.max_upload_bytes:
            raise UploadError(
                f"upload is {len(payload)} bytes; limit is {settings.max_upload_bytes}"
            )
        dataset_id = uuid.uuid4().hex
        safe_name = Path(filename).name or "upload.tif"
        tmp = self.tmp_dir / f"{dataset_id}-{safe_name}"
        final = self.dataset_path(dataset_id)
        try:
            with tmp.open("wb") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            meta = validate_raster(tmp)
            os.replace(tmp, final)
            record = meta.to_dict()
            record["id"] = dataset_id
            record["filename"] = safe_name
            with self._lock:
                data = self._load()
                data["datasets"][dataset_id] = record
                self._atomic_json(data)
            return DatasetMeta(**{**record, "bounds": tuple(record["bounds"])})
        except Exception:
            tmp.unlink(missing_ok=True)
            final.unlink(missing_ok=True)
            raise

    def get_dataset(self, dataset_id: str) -> DatasetMeta | None:
        data = self._load()
        record = data["datasets"].get(dataset_id)
        if record is None or not self.dataset_path(dataset_id).exists():
            return None
        record = dict(record)
        record["bounds"] = tuple(record["bounds"])
        return DatasetMeta(**record)

    def list_datasets(self) -> list[DatasetMeta]:
        data = self._load()
        result = []
        for record in data["datasets"].values():
            if self.dataset_path(record["id"]).exists():
                record = dict(record)
                record["bounds"] = tuple(record["bounds"])
                result.append(DatasetMeta(**record))
        return result

    def publish(self, version: str, dataset_ids: list[str]) -> list[DatasetMeta]:
        if not version or not isinstance(version, str):
            raise VersionError("version must be a non-empty string")
        if not dataset_ids:
            raise VersionError("dataset list must not be empty")
        if len(dataset_ids) != len(set(dataset_ids)):
            raise VersionError("dataset list contains duplicates")
        with self._lock:
            data = self._load()
            if version in data["versions"]:
                existing = data["versions"][version]["datasets"]
                if existing != dataset_ids:
                    raise VersionError(f"version {version!r} already exists")
                records = data["versions"][version]["records"]
            else:
                records = []
                for dataset_id in dataset_ids:
                    meta = self.get_dataset(dataset_id)
                    if meta is None:
                        raise VersionError(f"unknown dataset: {dataset_id}")
                    records.append(meta.to_dict())
                data["versions"][version] = {"datasets": dataset_ids, "records": records}
                self._atomic_json(data)
                records = data["versions"][version]["records"]
            result = []
            for record in records:
                record = dict(record)
                record["bounds"] = tuple(record["bounds"])
                result.append(DatasetMeta(**record))
            return result

    def version(self, version: str) -> list[DatasetMeta]:
        data = self._load()
        entry = data["versions"].get(version)
        if entry is None:
            raise VersionError(f"unknown version: {version}")
        result = []
        for record in entry["records"]:
            path = self.dataset_path(record["id"])
            if not path.exists():
                raise VersionError(f"published dataset is missing from storage: {record['id']}")
            record = dict(record)
            record["bounds"] = tuple(record["bounds"])
            result.append(DatasetMeta(**record))
        return result


registry = Registry()
