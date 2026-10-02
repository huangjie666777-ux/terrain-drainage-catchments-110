"""Windowed reprojection and priority mosaic construction."""

from __future__ import annotations

import math

import numpy as np
import rasterio
from pathlib import Path
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from rasterio import windows

from .storage import DatasetMeta

WEB_MERCATOR = "EPSG:3857"


def _source_window(src, dst_bounds: tuple[float, float, float, float], dst_crs=WEB_MERCATOR):
    left, bottom, right, top = transform_bounds(dst_crs, src.crs, *dst_bounds, densify_pts=21)
    left, bottom, right, top = (
        min(left, right),
        min(bottom, top),
        max(left, right),
        max(bottom, top),
    )
    window = windows.from_bounds(left, bottom, right, top, src.transform)
    # Round outward and pad: bilinear weights and a strict destination-validity
    # mask need every source cell touching the expanded target pixel area.
    row_off = math.floor(window.row_off) - 2
    col_off = math.floor(window.col_off) - 2
    row_end = math.ceil(window.row_off + window.height) + 2
    col_end = math.ceil(window.col_off + window.width) + 2
    row_off = max(row_off, 0)
    col_off = max(col_off, 0)
    row_end = min(row_end, src.height)
    col_end = min(col_end, src.width)
    if row_end <= row_off or col_end <= col_off:
        return None
    return windows.Window(col_off, row_off, col_end - col_off, row_end - row_off)


def _warp_dataset(
    path: str,
    dst_shape: tuple[int, int],
    dst_transform,
    dst_bounds: tuple[float, float, float, float],
    dst_crs=WEB_MERCATOR,
) -> np.ma.MaskedArray:
    """Read only overlapping source pixels and bilinearly warp elevations."""
    height, width = dst_shape
    elev = np.full(dst_shape, np.nan, dtype=np.float32)
    valid = np.zeros(dst_shape, dtype=np.uint8)
    with rasterio.open(path) as src:
        window = _source_window(src, dst_bounds, dst_crs)
        if window is None:
            return np.ma.array(elev, mask=True)
        source = src.read(1, window=window, masked=True)
        # getmaskarray also handles rasters with neither NoData nor an
        # explicit mask, where .mask collapses to the scalar np.False_.
        source_valid = (~np.ma.getmaskarray(source)).astype(np.uint8) * 255
        if not np.ma.is_masked(source):
            # Fully valid raster: pass a plain ndarray. A MaskedArray with a
            # scalar mask breaks rasterio.warp.reproject's shape handling.
            source = np.asarray(source.data)
        source_transform = src.window_transform(window)
        rasterio.warp.reproject(
            source=source,
            destination=elev,
            src_transform=source_transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
            num_threads=1,
        )
        rasterio.warp.reproject(
            source=source_valid,
            destination=valid,
            src_transform=source_transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=0,
            resampling=Resampling.nearest,
            num_threads=1,
        )
    # Only a fully valid bilinear kernel is accepted; this prevents NoData or
    # coverage edges from entering interpolation as a zero elevation.
    mask = (valid < 255) | ~np.isfinite(elev)
    return np.ma.array(elev, mask=mask)


def build_mosaic(
    datasets: list[DatasetMeta],
    dst_shape: tuple[int, int],
    dst_transform,
    dst_bounds: tuple[float, float, float, float],
    dataset_dir,
    dst_crs=WEB_MERCATOR,
) -> np.ma.MaskedArray:
    """Fill target cells using the first valid elevation in dataset order."""
    result = np.ma.array(np.full(dst_shape, np.nan, dtype=np.float32), mask=True)
    for dataset in datasets:
        if not np.any(result.mask):
            break
        candidate = _warp_dataset(
            str(Path(dataset_dir) / f"{dataset.id}.tif"),
            dst_shape,
            dst_transform,
            dst_bounds,
            dst_crs,
        )
        fill = result.mask & ~candidate.mask
        if fill.any():
            result[fill] = candidate[fill]
    return result
