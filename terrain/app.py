"""FastAPI HTTP interface for datasets, versions, tiles, and earthwork jobs."""

from __future__ import annotations

import threading

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from .config import ensure_dirs, settings
from .earthwork import EarthworkError
from .jobs import JobManager
from .storage import Registry, UploadError, VersionError
from .tiles import TileCache, TileError, TileRenderer

ensure_dirs()
registry = Registry()
renderer = TileRenderer(
    dataset_dir=str(registry.dataset_dir),
    cache=TileCache(),
    slots=threading.BoundedSemaphore(settings.max_concurrent_jobs),
)
job_manager = JobManager(registry)
app = FastAPI(title="Terrain Hillshade Tiles", version="1.1.0")


class PublishRequest(BaseModel):
    version: str = Field(min_length=1, max_length=128)
    datasets: list[str] = Field(min_length=1, max_length=512)


class EarthworkRequest(BaseModel):
    before_version: str = Field(min_length=1, max_length=128)
    after_version: str = Field(min_length=1, max_length=128)
    region: dict
    epsg: int
    resolution: float = Field(gt=0)


@app.exception_handler(UploadError)
def upload_handler(_request, exc: UploadError):
    return JSONResponse(status_code=400, content={"error": str(exc)})


@app.exception_handler(VersionError)
def version_handler(_request, exc: VersionError):
    status_code = 409 if "already exists" in str(exc) else 400
    if str(exc).startswith("unknown version") or "missing from storage" in str(exc):
        status_code = 404
    return JSONResponse(status_code=status_code, content={"error": str(exc)})


@app.exception_handler(TileError)
def tile_handler(_request, exc: TileError):
    return JSONResponse(status_code=400, content={"error": str(exc)})


@app.exception_handler(EarthworkError)
def earthwork_handler(_request, exc: EarthworkError):
    return JSONResponse(status_code=400, content={"error": str(exc)})


@app.get("/health")
def health():
    return {
        "status": "ok",
        "max_upload_bytes": settings.max_upload_bytes,
        "max_zoom": settings.max_zoom,
        "max_concurrent_jobs": settings.max_concurrent_jobs,
        "max_grid_cells": settings.max_grid_cells,
        "cache_tiles": len(renderer.cache),
    }


@app.post("/datasets")
def upload_dataset(file: UploadFile = File(...)):
    payload = file.file.read(settings.max_upload_bytes + 1)
    if len(payload) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="file exceeds upload size limit")
    if not payload:
        raise HTTPException(status_code=400, detail="empty upload")
    try:
        meta = registry.ingest(file.filename or "upload.tif", payload)
    except UploadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"id": meta.id, "metadata": meta.to_dict() | {"id": meta.id}}


@app.get("/datasets")
def list_datasets():
    items = [item.to_dict() for item in registry.list_datasets()]
    return {"datasets": items}


@app.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: str):
    meta = registry.get_dataset(dataset_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="dataset not found")
    return {"id": meta.id, "metadata": meta.to_dict() | {"id": meta.id}}


@app.post("/versions")
def publish_version(request: PublishRequest):
    try:
        items = registry.publish(request.version, request.datasets)
    except VersionError as exc:
        if "already exists" in str(exc):
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "version": request.version,
        "datasets": [item.to_dict() for item in items],
        "dataset_ids": [item.id for item in items],
    }


@app.get("/versions/{version}")
def get_version(version: str):
    try:
        items = registry.version(version)
    except VersionError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "version": version,
        "dataset_ids": [item.id for item in items],
        "datasets": [item.to_dict() for item in items],
    }


@app.get("/tiles/{version}/{z}/{x}/{y}.png")
def tile(
    version: str,
    z: int,
    x: int,
    y: int,
    azimuth: float = Query(default=settings.default_azimuth),
    altitude: float = Query(default=settings.default_altitude),
):
    try:
        datasets = registry.version(version)
        png = renderer.render(version, datasets, z, x, y, azimuth, altitude)
    except VersionError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except TileError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return Response(
        content=png,
        media_type="image/png",
        headers={"Cache-Control": "private, max-age=3600"},
    )


@app.post("/earthwork/jobs", status_code=201)
def submit_earthwork(request: EarthworkRequest):
    try:
        return job_manager.submit(
            request.before_version,
            request.after_version,
            request.region,
            request.epsg,
            request.resolution,
        )
    except VersionError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except EarthworkError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/earthwork/jobs/{job_id}")
def get_earthwork_job(job_id: str):
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job


@app.post("/earthwork/jobs/{job_id}/cancel")
def cancel_earthwork_job(job_id: str):
    job = job_manager.cancel(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job["status"] not in ("queued", "running"):
        raise HTTPException(
            status_code=409, detail=f"job is {job['status']}; cannot cancel"
        )
    return job_manager.get(job_id)


@app.get("/earthwork/jobs/{job_id}/diff.tif")
def download_earthwork_diff(job_id: str):
    if job_manager.get(job_id) is None:
        raise HTTPException(status_code=404, detail="job not found")
    path = job_manager.diff_path(job_id)
    if path is None:
        raise HTTPException(status_code=409, detail="job has no completed diff raster")
    return FileResponse(path, media_type="image/tiff", filename="diff.tif")
