from __future__ import annotations

import io
import json
import threading
import time

import numpy as np
import pytest
import rasterio
from fastapi.testclient import TestClient

import terrain.jobs as jobs_mod
from terrain import app as app_module
from terrain.earthwork import DIFF_NODATA
from terrain.jobs import JobManager

EPSG = 32647
BOUNDS = (500_000.0, 2_950_000.0, 500_200.0, 2_950_200.0)  # 200 m x 200 m
RESOLUTION = 10.0


def region_geojson() -> dict:
    shell = [
        [500_020.0, 2_950_020.0],
        [500_180.0, 2_950_020.0],
        [500_180.0, 2_950_180.0],
        [500_020.0, 2_950_180.0],
        [500_020.0, 2_950_020.0],
    ]
    hole = [
        [500_080.0, 2_950_080.0],
        [500_120.0, 2_950_080.0],
        [500_120.0, 2_950_120.0],
        [500_080.0, 2_950_120.0],
        [500_080.0, 2_950_080.0],
    ]
    return {"type": "Polygon", "coordinates": [shell, hole]}


def epoch_surfaces():
    # 20x20 source pixels of 10 m on the same alignment as the 10 m target
    # grid, so reprojection is the identity and volumes are exact.
    cols = np.arange(20, dtype=np.float32)
    before = 100.0 + 0.01 * cols[None, :] * 10.0  # linear in x
    before = np.repeat(before, 20, axis=0).astype(np.float32)
    after = before + 2.0
    after[4:8, 12:16] -= 5.0  # cut block: net -3 m over 40 m x 40 m
    after[12:16, 4:8] = -9999.0  # survey gap: 40 m x 40 m inside the region
    return before, after


@pytest.fixture()
def service(temp_service, make_tif, tmp_path, monkeypatch):
    registry, renderer, _tmp = temp_service
    before, after = epoch_surfaces()
    ids = {}
    for name, values in (("before", before), ("after", after)):
        path = tmp_path / f"{name}.tif"
        make_tif(path, values, crs=f"EPSG:{EPSG}", bounds=BOUNDS, write_mask=True)
        ids[name] = registry.ingest(path.name, path.read_bytes()).id
    registry.publish("epoch-before", [ids["before"]])
    registry.publish("epoch-after", [ids["after"]])
    manager = JobManager(registry, root=tmp_path / "jobs", max_workers=1)
    monkeypatch.setattr(app_module, "registry", registry)
    monkeypatch.setattr(app_module, "renderer", renderer)
    monkeypatch.setattr(app_module, "job_manager", manager)
    return TestClient(app_module.app), manager, registry


def submit_payload(**overrides):
    payload = {
        "before_version": "epoch-before",
        "after_version": "epoch-after",
        "region": region_geojson(),
        "epsg": EPSG,
        "resolution": RESOLUTION,
    }
    payload.update(overrides)
    return payload


def wait_for_job(client, job_id, timeout=30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/earthwork/jobs/{job_id}").json()
        if job["status"] not in ("queued", "running"):
            return job
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish: {job}")


def test_submit_rejects_bad_inputs(service):
    client, _manager, _registry = service
    assert client.post("/earthwork/jobs", json=submit_payload(epsg=4326)).status_code == 400
    assert (
        client.post("/earthwork/jobs", json=submit_payload(before_version="nope")).status_code
        == 404
    )
    bowtie = {
        "type": "Polygon",
        "coordinates": [[[500020.0, 2950020.0], [500180.0, 2950180.0],
                         [500180.0, 2950020.0], [500020.0, 2950180.0],
                         [500020.0, 2950020.0]]],
    }
    assert client.post("/earthwork/jobs", json=submit_payload(region=bowtie)).status_code == 400
    nan_region = region_geojson()
    nan_region["coordinates"][0][1] = [500180.0, float("nan")]
    response = client.post(
        "/earthwork/jobs",
        content=json.dumps(submit_payload(region=nan_region)),
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 400
    assert client.post("/earthwork/jobs", json=submit_payload(resolution=-5)).status_code == 422
    assert client.get("/earthwork/jobs/does-not-exist").status_code == 404


def test_grid_size_limit(service, monkeypatch):
    client, _manager, _registry = service
    import dataclasses

    import terrain.earthwork as earthwork_mod

    limited = dataclasses.replace(earthwork_mod.settings, max_grid_cells=100)
    monkeypatch.setattr(earthwork_mod, "settings", limited)
    response = client.post("/earthwork/jobs", json=submit_payload())
    assert response.status_code == 400
    assert "limit" in response.json()["detail"]


def test_cut_fill_volumes_and_diff_download(service):
    client, manager, _registry = service
    response = client.post("/earthwork/jobs", json=submit_payload())
    assert response.status_code == 201, response.text
    job_id = response.json()["id"]
    job = wait_for_job(client, job_id)
    assert job["status"] == "completed", job["error"]

    result = job["result"]
    # Region: 160x160 m minus a 40x40 m hole = 24000 m2. The after-epoch
    # NoData block removes another 1600 m2 of measured area.
    assert result["total_area_m2"] == pytest.approx(24_000.0)
    assert result["missing_area_m2"] == pytest.approx(1_600.0)
    assert result["valid_area_m2"] == pytest.approx(22_400.0)
    assert result["coverage"] == pytest.approx(22_400.0 / 24_000.0)
    # Cut: 1600 m2 at -3 m. Fill: +2 m over the remaining valid area.
    assert result["cut_volume_m3"] == pytest.approx(4_800.0)
    assert result["fill_volume_m3"] == pytest.approx(2.0 * (22_400.0 - 1_600.0))
    assert result["net_volume_m3"] == pytest.approx(
        result["fill_volume_m3"] - result["cut_volume_m3"]
    )
    assert job["grid"]["width"] == 16 and job["grid"]["height"] == 16

    download = client.get(f"/earthwork/jobs/{job_id}/diff.tif")
    assert download.status_code == 200
    with rasterio.io.MemoryFile(download.content) as memfile:
        with memfile.open() as src:
            assert src.crs.to_epsg() == EPSG
            assert src.nodata == DIFF_NODATA
            diff = src.read(1)
    assert diff[0, 0] == pytest.approx(2.0)  # fill cell
    assert diff[5, 10] == pytest.approx(-3.0)  # cut cell
    assert diff[10, 2] == DIFF_NODATA  # missing in after epoch

    # Results survive a manager restart and stay downloadable.
    restarted = JobManager(manager.registry, root=manager.root, max_workers=1)
    assert restarted.get(job_id)["status"] == "completed"
    assert restarted.diff_path(job_id) is not None


def test_cancel_leaves_no_partial_result(service, monkeypatch):
    client, manager, _registry = service
    started = threading.Event()
    release = threading.Event()
    original = jobs_mod.run_earthwork

    def slow_run(*args, **kwargs):
        started.set()
        assert release.wait(10)
        return original(*args, **kwargs)

    monkeypatch.setattr(jobs_mod, "run_earthwork", slow_run)
    job_id = client.post("/earthwork/jobs", json=submit_payload()).json()["id"]
    assert started.wait(10)
    cancel = client.post(f"/earthwork/jobs/{job_id}/cancel")
    assert cancel.status_code == 200
    release.set()
    job = wait_for_job(client, job_id)
    assert job["status"] == "cancelled"
    assert manager.diff_path(job_id) is None
    assert not (manager.root / job_id).exists()
    assert client.get(f"/earthwork/jobs/{job_id}/diff.tif").status_code == 409
    assert client.post(f"/earthwork/jobs/{job_id}/cancel").status_code == 409


def test_restart_marks_unfinished_jobs_interrupted(service, tmp_path):
    _client, manager, _registry = service
    root = manager.root
    manager.shutdown()
    jobs = json.loads((root / "jobs.json").read_text()) if (root / "jobs.json").exists() else {}
    jobs["stuck"] = {
        "id": "stuck",
        "status": "running",
        "error": None,
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    (root / "jobs.json").write_text(json.dumps(jobs))
    restarted = JobManager(manager.registry, root=root, max_workers=1)
    stuck = restarted.get("stuck")
    assert stuck["status"] == "interrupted"
    assert restarted.diff_path("stuck") is None
