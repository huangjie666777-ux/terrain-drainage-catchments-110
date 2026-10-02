"""Runtime configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value is not None else default


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path(os.environ.get("TERRAIN_DATA_DIR", "data"))
    max_upload_bytes: int = _int("TERRAIN_MAX_UPLOAD_BYTES", 64 * 1024 * 1024)
    tile_size: int = 256
    max_zoom: int = 18
    max_concurrent_jobs: int = _int("TERRAIN_MAX_CONCURRENT_JOBS", 2)
    cache_max_tiles: int = _int("TERRAIN_CACHE_MAX_TILES", 256)
    max_grid_cells: int = _int("TERRAIN_MAX_GRID_CELLS", 4_000_000)
    earthwork_chunk_rows: int = _int("TERRAIN_EARTHWORK_CHUNK_ROWS", 256)
    default_azimuth: float = 315.0
    default_altitude: float = 45.0


settings = Settings()


def ensure_dirs() -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    (settings.data_dir / "datasets").mkdir(exist_ok=True)
    (settings.data_dir / "tmp").mkdir(exist_ok=True)
