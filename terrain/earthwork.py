"""Cut/fill volume accounting between two published terrain versions.

The construction region is a Polygon/MultiPolygon in a UTM metre CRS. The
output grid is the region bounding box snapped outward to integer multiples
of the requested resolution. Both epochs are bilinearly reprojected onto
the same grid cell centers with the published priority hole-filling order,
and each cell is weighted by its exact intersection area with the region
(holes excluded), not by a center-in/out test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import rasterio
import shapely
from affine import Affine
from rasterio.crs import CRS
from rasterio.windows import Window
from shapely.geometry import shape as geojson_shape
from shapely.geometry.base import BaseGeometry

from .config import settings
from .mosaic import build_mosaic
from .storage import DatasetMeta, is_utm_crs

DIFF_NODATA = -9999.0


class EarthworkError(ValueError):
    """Raised when an earthwork request or its inputs are invalid."""


class JobCancelled(Exception):
    """Raised inside a worker when the job cancel event is set."""


def validate_utm_epsg(epsg: int) -> CRS:
    """Accept only projected UTM metre CRS codes."""
    try:
        crs = CRS.from_epsg(int(epsg))
    except (ValueError, TypeError) as exc:
        raise EarthworkError(f"unknown EPSG code: {epsg!r}") from exc
    if crs is None:
        raise EarthworkError(f"unknown EPSG code: {epsg!r}")
    if not is_utm_crs(crs):
        raise EarthworkError(f"EPSG:{epsg} is not a UTM metre projected CRS")
    return crs


def validate_region(geometry: Any) -> BaseGeometry:
    """Validate a GeoJSON Polygon/MultiPolygon in UTM metre coordinates."""
    if not isinstance(geometry, dict):
        raise EarthworkError("region must be a GeoJSON Polygon or MultiPolygon object")
    if geometry.get("type") not in ("Polygon", "MultiPolygon"):
        raise EarthworkError("region must be a GeoJSON Polygon or MultiPolygon")
    try:
        geom = geojson_shape(geometry)
    except (ValueError, TypeError, KeyError) as exc:
        raise EarthworkError(f"region geometry is malformed: {exc}") from exc
    if geom.is_empty or not math.isfinite(geom.area) or geom.area <= 0.0:
        raise EarthworkError("region geometry is empty or has zero area")
    coords = shapely.get_coordinates(geom)
    if coords.size == 0 or not np.isfinite(coords).all():
        raise EarthworkError("region coordinates must all be finite numbers")
    if not geom.is_valid:
        raise EarthworkError(
            f"region geometry is invalid (self-intersecting or degenerate): "
            f"{shapely.explain_validity(geom)}"
        )
    return geom


@dataclass(frozen=True)
class GridSpec:
    epsg: int
    resolution: float
    transform: Affine
    width: int
    height: int

    @property
    def crs(self) -> str:
        return f"EPSG:{self.epsg}"

    @property
    def cell_area(self) -> float:
        return self.resolution * self.resolution

    @property
    def cells(self) -> int:
        return self.width * self.height

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        left = self.transform.c
        top = self.transform.f
        return (
            left,
            top - self.height * self.resolution,
            left + self.width * self.resolution,
            top,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "epsg": self.epsg,
            "resolution": self.resolution,
            "width": self.width,
            "height": self.height,
            "cells": self.cells,
            "bounds": list(self.bounds),
        }


def aligned_grid(region: BaseGeometry, resolution: float, epsg: int) -> GridSpec:
    """Snap the region bounding box outward to resolution multiples."""
    if not isinstance(resolution, (int, float)) or not math.isfinite(resolution) or resolution <= 0:
        raise EarthworkError("resolution must be a positive finite number of metres")
    minx, miny, maxx, maxy = region.bounds
    # Plausibility guard: region coordinates must look like UTM metres.
    if not (100_000.0 <= minx <= 900_000.0 and 100_000.0 <= maxx <= 900_000.0):
        raise EarthworkError("region easting is outside the plausible UTM metre range")
    if not (0.0 <= miny <= 10_000_000.0 and 0.0 <= maxy <= 10_000_000.0):
        raise EarthworkError("region northing is outside the plausible UTM metre range")
    left = math.floor(minx / resolution) * resolution
    right = math.ceil(maxx / resolution) * resolution
    bottom = math.floor(miny / resolution) * resolution
    top = math.ceil(maxy / resolution) * resolution
    width = int(round((right - left) / resolution))
    height = int(round((top - bottom) / resolution))
    if width < 1 or height < 1:
        raise EarthworkError("aligned grid is empty for the given region and resolution")
    if width * height > settings.max_grid_cells:
        raise EarthworkError(
            f"aligned grid is {width}x{height} = {width * height} cells; "
            f"limit is {settings.max_grid_cells}"
        )
    transform = Affine(resolution, 0.0, left, 0.0, -resolution, top)
    return GridSpec(epsg=epsg, resolution=float(resolution), transform=transform,
                    width=width, height=height)


def chunk_weights(
    region: BaseGeometry, grid: GridSpec, row_off: int, rows: int
) -> np.ndarray:
    """Exact intersection area of each grid cell with the region, in m2."""
    res = grid.resolution
    left = grid.transform.c
    top = grid.transform.f
    cols = np.arange(grid.width, dtype=np.float64)
    row_idx = row_off + np.arange(rows, dtype=np.float64)
    xmin = left + cols * res
    ymax = top - row_idx * res
    xx, yy = np.meshgrid(xmin, ymax)
    cells = shapely.box(xx, yy - res, xx + res, yy)
    weights = np.zeros((rows, grid.width), dtype=np.float64)
    covered = shapely.covered_by(cells, region)
    weights[covered] = grid.cell_area
    remainder = ~covered
    if remainder.any():
        subset = cells[remainder]
        hits = shapely.intersects(subset, region)
        if hits.any():
            partial = shapely.area(shapely.intersection(subset[hits], region))
            weights[remainder] = 0.0
            weights[np.where(remainder)[0][hits], np.where(remainder)[1][hits]] = partial
    return weights


def run_earthwork(
    before: list[DatasetMeta],
    after: list[DatasetMeta],
    grid: GridSpec,
    region: BaseGeometry,
    dataset_dir: str,
    diff_path,
    cancel_event=None,
    on_progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Compute cut/fill volumes chunk by chunk and write the diff GeoTIFF.

    Only cells valid in both epochs are measured; other in-region cells add
    to the missing area and are written as NoData, never as zero difference.
    """
    chunk_rows = max(1, min(settings.earthwork_chunk_rows, grid.height))
    profile = {
        "driver": "GTiff",
        "width": grid.width,
        "height": grid.height,
        "count": 1,
        "dtype": "float32",
        "crs": grid.crs,
        "transform": grid.transform,
        "nodata": DIFF_NODATA,
        "compress": "deflate",
    }
    total_area = 0.0
    valid_area = 0.0
    missing_area = 0.0
    fill_volume = 0.0
    cut_volume = 0.0
    with rasterio.open(diff_path, "w", **profile) as dst:
        for row_off in range(0, grid.height, chunk_rows):
            if cancel_event is not None and cancel_event.is_set():
                raise JobCancelled()
            rows = min(chunk_rows, grid.height - row_off)
            chunk_shape = (rows, grid.width)
            transform = grid.transform * Affine.translation(0, row_off)
            bounds = (
                grid.transform.c,
                transform.f - rows * grid.resolution,
                grid.transform.c + grid.width * grid.resolution,
                transform.f,
            )
            before_arr = build_mosaic(
                before, chunk_shape, transform, bounds, dataset_dir, dst_crs=grid.crs
            )
            after_arr = build_mosaic(
                after, chunk_shape, transform, bounds, dataset_dir, dst_crs=grid.crs
            )
            weights = chunk_weights(region, grid, row_off, rows)
            both_valid = ~np.ma.getmaskarray(before_arr) & ~np.ma.getmaskarray(after_arr)
            diff = np.full(chunk_shape, DIFF_NODATA, dtype=np.float32)
            if both_valid.any():
                delta = (
                    after_arr.data[both_valid].astype(np.float64)
                    - before_arr.data[both_valid].astype(np.float64)
                )
                diff[both_valid] = delta.astype(np.float32)
                w = weights[both_valid]
                positive = delta > 0.0
                fill_volume += float(np.sum(delta[positive] * w[positive]))
                negative = delta < 0.0
                cut_volume += float(np.sum(-delta[negative] * w[negative]))
                valid_area += float(np.sum(w))
            missing = ~both_valid
            if missing.any():
                missing_area += float(np.sum(weights[missing]))
            total_area += float(np.sum(weights))
            dst.write(diff, 1, window=Window(0, row_off, grid.width, rows))
            if on_progress is not None:
                on_progress(row_off + rows, grid.height)
    return {
        "total_area_m2": total_area,
        "valid_area_m2": valid_area,
        "missing_area_m2": missing_area,
        "coverage": (valid_area / total_area) if total_area > 0 else 0.0,
        "fill_volume_m3": fill_volume,
        "cut_volume_m3": cut_volume,
        "net_volume_m3": fill_volume - cut_volume,
        "grid": grid.to_dict(),
    }
