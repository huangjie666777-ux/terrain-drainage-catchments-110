#!/usr/bin/env python3
"""Generate before/after UTM slope rasters with holes plus a region GeoJSON.

Both epochs share the same 2 m grid on a synthetic hillside in EPSG:32647.
The "before" surface is a smooth slope with a NoData + explicit-mask hole;
the "after" surface adds a raised fill platform, an excavated cut pit, and
its own NoData hole elsewhere. The construction region is a polygon with an
interior hole, so hole exclusion and partial-cell weighting are exercised.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

EPSG = 32647
WEST, NORTH = 500_000.0, 2_950_400.0
RES = 2.0
WIDTH = HEIGHT = 150  # 300 m x 300 m
NODATA = -9999.0


def base_surface() -> np.ndarray:
    cols, rows = np.meshgrid(np.arange(WIDTH), np.arange(HEIGHT))
    x = cols * RES
    y = (HEIGHT - rows) * RES
    slope = 1200.0 + 0.35 * x - 0.20 * y
    ridge = 18.0 * np.exp(-(((x - 170.0) ** 2) / 4800.0 + ((y - 150.0) ** 2) / 9000.0))
    return (slope + ridge).astype(np.float32)


def write(path: Path, values: np.ndarray, with_mask: bool) -> None:
    profile = {
        "driver": "GTiff",
        "width": WIDTH,
        "height": HEIGHT,
        "count": 1,
        "dtype": "float32",
        "crs": f"EPSG:{EPSG}",
        "transform": from_origin(WEST, NORTH, RES, RES),
        "nodata": NODATA,
        "compress": "deflate",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(values, 1)
        if with_mask:
            dst.write_mask(np.isfinite(values) & (values != NODATA))


def region_geojson() -> dict:
    # 220 m x 260 m site with a 40 m x 40 m excluded hole; edges sit off the
    # 2 m raster grid so partial-cell area weighting is exercised. The site
    # spans the fill platform, the cut pit, and both epochs' NoData holes.
    x0, y0, x1, y1 = 500_041.0, 2_950_111.0, 500_261.0, 2_950_371.0
    hx0, hy0, hx1, hy1 = 500_141.0, 2_950_121.0, 500_181.0, 2_950_161.0
    return {
        "type": "Polygon",
        "coordinates": [
            [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]],
            [[hx0, hy0], [hx1, hy0], [hx1, hy1], [hx0, hy1], [hx0, hy0]],
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("samples/earthwork"))
    args = parser.parse_args()

    before = base_surface()
    # NoData + explicit mask hole in the before epoch (rows 40:60, cols 30:55).
    before[40:60, 30:55] = NODATA
    write(args.out / "before_slope_with_hole.tif", before, with_mask=True)

    after = base_surface()
    # Fill platform: +2.5 m over cols 60:100, rows 20:50.
    after[20:50, 60:100] += 2.5
    # Cut pit: -3.0 m over cols 105:135, rows 90:125.
    after[90:125, 105:135] -= 3.0
    # A fresh survey gap in the after epoch (rows 100:115, cols 20:35).
    after[100:115, 20:35] = NODATA
    write(args.out / "after_cut_fill_with_hole.tif", after, with_mask=True)

    region_path = args.out / "region.geojson"
    with region_path.open("w", encoding="utf-8") as fh:
        json.dump(region_geojson(), fh, indent=2)

    print(f"wrote {args.out / 'before_slope_with_hole.tif'}")
    print(f"wrote {args.out / 'after_cut_fill_with_hole.tif'}")
    print(f"wrote {region_path}")
    print(f"publish both as two versions, then POST /earthwork/jobs with epsg={EPSG}")


if __name__ == "__main__":
    main()
