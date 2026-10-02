# Terrain Hillshade Tile + Earthwork Service

Pure-backend FastAPI service for uploading single-band elevation GeoTIFFs, serving 256×256 XYZ grayscale RGBA PNG hillshade tiles in EPSG:3857, and computing cut/fill (挖填方) volumes between two published terrain versions over a construction region.

## Modules

- `terrain/storage.py` — atomic GeoTIFF ingestion, validation, dataset metadata, and immutable published versions.
- `terrain/mosaic.py` — bounded source-window reads, reprojection to Web Mercator or UTM pixel centers, bilinear elevation sampling, and ordered hole filling.
- `terrain/hillshade.py` — Horn 3×3 gradients on the stitched one-pixel-padded grid, Lambert diffuse shading, and latitude-corrected ground spacing.
- `terrain/tiles.py` — XYZ validation, 258×258 render grid, 256×256 crop, bounded render semaphore, and bounded LRU PNG cache.
- `terrain/earthwork.py` — UTM/region/grid validation, resolution-aligned grid construction, exact per-cell region intersection weights, and chunked cut/fill integration with diff GeoTIFF output.
- `terrain/jobs.py` — persistent background earthwork jobs: submit/query/cancel, bounded worker pool, atomic result promotion, and restart recovery.
- `terrain/app.py` — HTTP API.
- `scripts/generate_samples.py` — creates a slope raster with a NoData/explicit-mask hole and an overlapping lower-priority fill raster.
- `scripts/generate_earthwork_samples.py` — creates before/after UTM slope rasters with holes plus a construction-region GeoJSON with an interior hole.

## Run

```bash
.venv/bin/uvicorn terrain.app:app --host 127.0.0.1 --port 8000
```

Persistent files default to `data/`: GeoTIFFs are in `data/datasets/`, temporary uploads in `data/tmp/`, and dataset/version metadata in `data/registry.json`. The registry is read at startup, so datasets and versions survive a process restart.

Generate the sample pair:

```bash
.venv/bin/python scripts/generate_samples.py --out samples
```

Generate the two-epoch earthwork sample set (UTM EPSG:32647, 2 m grid, holes in both epochs, region with an interior hole):

```bash
.venv/bin/python scripts/generate_earthwork_samples.py --out samples/earthwork
```

## HTTP Protocol

### Upload one raster

```bash
curl -F 'file=@samples/01_slope_with_hole.tif;type=image/tiff' \
  http://127.0.0.1:8000/datasets
```

The response contains `id` and persistent metadata. Uploads must be single-band GeoTIFF elevation rasters in metres, with CRS `EPSG:4326` or `EPSG:3857`, and a valid non-identity affine transform. Unknown CRS, multiple bands, missing georeferencing, zero/non-invertible transforms, empty files, oversize uploads, and corrupt files are rejected. The file is validated in a temporary path and atomically renamed only after success; failed attempts leave no dataset.

### Publish an ordered, immutable version

```bash
curl -sS -X POST http://127.0.0.1:8000/versions \
  -H 'Content-Type: application/json' \
  -d '{"version":"demo-v1","datasets":["PRIMARY_ID","FILL_ID"]}'
```

Dataset order is priority order. For each target pixel, the first valid elevation in the ordered list is used. NoData and mask holes in a higher-priority dataset may therefore be filled by a lower-priority dataset. Duplicate IDs and unknown IDs are rejected. A version cannot be overwritten with a different ordered list; repeating the identical list is an idempotent success. Reads must name an existing version and use the immutable records snapshot for that request.

### Fetch a tile

```bash
curl -fL 'http://127.0.0.1:8000/tiles/demo-v1/12/3184/1730.png?azimuth=315&altitude=45' \
  -o tile.png
```

- Path: `/tiles/{version}/{z}/{x}/{y}.png`.
- Zoom range: `0 <= z <= 18`; `x` and `y` are standard XYZ tile coordinates and are range-checked for the zoom.
- `azimuth`: degrees clockwise from north, `[0, 360)`; default `315`.
- `altitude`: degrees above the horizon, `(0, 90]`; default `45`.
- Output: 256×256, 8-bit grayscale replicated into RGB plus alpha, PNG.
- Outside source coverage the alpha channel is zero. NoData is never treated as zero elevation or allowed into bilinear interpolation: a target center is accepted only when its bilinear source kernel is valid.
- A 1-pixel border is rendered before mosaicking/gradients, then removed. A tile pixel is transparent unless all nine Horn neighbours are valid after priority fill, avoiding artificial cliffs at holes.
- Horizontal spacing uses Web Mercator projected pixel size multiplied by `cos(latitude)`, not raw projected metres.

Other endpoints:

- `GET /health`
- `GET /datasets`
- `GET /datasets/{id}`
- `GET /versions/{version}`

## Earthwork Cut/Fill Accounting

`POST /earthwork/jobs` submits a background job comparing two published versions over a construction region:

```bash
curl -sS -X POST http://127.0.0.1:8000/earthwork/jobs \
  -H 'Content-Type: application/json' \
  -d '{
        "before_version": "epoch-2024",
        "after_version": "epoch-2025",
        "region": '"$(cat samples/earthwork/region.geojson)"',
        "epsg": 32647,
        "resolution": 2.0
      }'
```

Request rules:

- `before_version` / `after_version` must be existing published versions (unknown → 404). The ordered dataset lists of both versions are frozen into the job record at submission time.
- `region` is a GeoJSON `Polygon` or `MultiPolygon` (holes allowed) whose coordinates are metres in the UTM CRS given by `epsg`. Self-intersecting/degenerate polygons, non-finite coordinates, non-Polygon types, and implausible UTM coordinate ranges are rejected (400).
- `epsg` must be a projected UTM metre CRS (e.g. `32647`); geographic or non-UTM codes are rejected (400).
- `resolution` is a positive finite grid size in metres. The region bounding box is snapped outward to integer multiples of the resolution; grids larger than `TERRAIN_MAX_GRID_CELLS` cells are rejected (400).
- Both epochs must share the same vertical datum; elevations are differenced directly.

Computation:

- Both epochs are bilinearly reprojected onto the same aligned UTM grid cell centers, each using its published priority order to fill NoData/mask holes. Reading and integration proceed in row chunks (`TERRAIN_EARTHWORK_CHUNK_ROWS`); the full survey is never loaded into memory.
- Each cell is weighted by its exact intersection area with the region (Shapely; holes subtracted) — not by a center-in/out test — so boundary cells contribute partial areas.
- `diff = after - before` per cell. Positive differences accumulate fill (填方), negative differences accumulate cut (挖方) as absolute amounts; `net = fill - cut`.
- Only cells valid in **both** epochs are measured. In-region cells missing in either epoch add to `missing_area_m2` and are written as NoData — they are never treated as zero change.

Job lifecycle endpoints:

```bash
curl -sS http://127.0.0.1:8000/earthwork/jobs/JOB_ID            # status + result
curl -sS -X POST http://127.0.0.1:8000/earthwork/jobs/JOB_ID/cancel
curl -fL http://127.0.0.1:8000/earthwork/jobs/JOB_ID/diff.tif -o diff.tif
```

- Statuses: `queued → running → completed | failed | cancelled`; `interrupted` marks jobs left active across a service restart. Cancelling a non-active job returns 409; downloading a diff for a non-completed job returns 409.
- The result JSON reports `total_area_m2`, `valid_area_m2`, `missing_area_m2`, `coverage`, `fill_volume_m3`, `cut_volume_m3`, and `net_volume_m3`.
- The diff GeoTIFF keeps the UTM spatial reference (`EPSG:{epsg}`) and writes unmeasured cells as NoData (`-9999`).
- Workers write into a temporary directory and atomically promote results on success, so failed or cancelled jobs never publish partial output. Jobs and results persist under `<data_dir>/earthwork/` (`jobs.json` plus one directory per completed job); after a restart, unfinished jobs are marked `interrupted` and completed results stay downloadable.
- At most `TERRAIN_MAX_CONCURRENT_JOBS` earthwork jobs run simultaneously; the rest stay queued.

### Grid approximation notes

Elevations are sampled once per output cell center after bilinear reprojection, so sub-cell terrain roughness is approximated by the cell-center value — a finer `resolution` reduces this error. GDAL's bilinear kernel slightly smooths sharp elevation steps when source and target grids are not aligned; volumes on exactly aligned grids are exact. Region-area weights themselves are exact polygon intersections, not rasterized approximations.

## Limits and Configuration

Environment variables:

| Variable | Default | Meaning |
| --- | ---: | --- |
| `TERRAIN_DATA_DIR` | `data` | persistent dataset and registry directory |
| `TERRAIN_MAX_UPLOAD_BYTES` | `67108864` | accepted upload size limit (64 MiB) |
| `TERRAIN_MAX_CONCURRENT_JOBS` | `2` | simultaneous tile render jobs |
| `TERRAIN_CACHE_MAX_TILES` | `256` | in-memory LRU PNG tile capacity |
| `TERRAIN_MAX_ZOOM` | `18` | maximum supported XYZ zoom |
| `TERRAIN_MAX_GRID_CELLS` | `4000000` | maximum aligned earthwork grid size |
| `TERRAIN_EARTHWORK_CHUNK_ROWS` | `256` | rows per earthwork read/compute chunk |

The cache key is `(version, z, x, y, azimuth, altitude)`. Tile rendering and earthwork integration never read a complete source raster: each source is intersected with the expanded target bounds and read through a Rasterio window. Uploads accept EPSG:4326, EPSG:3857, and UTM projected single-band rasters.

## Tests and Build Check

```bash
.venv/bin/python -m pytest
.venv/bin/python -m compileall -q terrain scripts tests
```

Tests cover persistence/restart, invalid uploads and cleanup, immutable ordered versions, transparent no-coverage output, ordered hole fill, Horn-neighbour transparency, lighting-sensitive output, cache-key separation, rendering of fully valid rasters with neither NoData nor mask, earthwork request validation (non-UTM CRS, self-intersecting/NaN regions, unknown versions, grid-size limit), exact cut/fill volumes with hole and missing-area accounting, diff GeoTIFF CRS/NoData download, cancel without partial results, and restart interruption.
