from __future__ import annotations

import numpy as np
import rasterio
from affine import Affine
import pytest

from terrain.storage import UploadError, VersionError


def elevation(width=80, height=80):
    x = np.linspace(0, 1, width, dtype=np.float32)
    y = np.linspace(0, 1, height, dtype=np.float32)
    return (1000 + 300 * x[None, :] - 200 * y[:, None]).astype(np.float32)


def test_valid_upload_persists_and_reloads(temp_service, make_tif):
    registry, _renderer, tmp = temp_service
    source = tmp / "source.tif"
    make_tif(source, elevation(), crs="EPSG:3857", bounds=(-1000, 1000, 1000, 3000))
    meta = registry.ingest("dem.tif", source.read_bytes())
    assert registry.dataset_path(meta.id).exists()

    from terrain.storage import Registry

    reloaded = Registry(root=registry.root)
    assert reloaded.get_dataset(meta.id).crs == "EPSG:3857"


def test_rejects_multiband_unknown_crs_bad_transform_and_corrupt(temp_service, make_tif, tmp_path):
    registry, _renderer, _tmp = temp_service
    multi = np.stack([elevation(), elevation()])
    source = tmp_path / "multi.tif"
    make_tif(source, multi, crs="EPSG:3413", count=2)
    with pytest.raises(UploadError) as err:
        registry.ingest("multi.tif", source.read_bytes())
    exc = err.value
    assert "EPSG" in str(exc)
    assert not list(registry.dataset_dir.glob("*.tif"))

    bad = tmp_path / "bad.tif"
    make_tif(bad, elevation(), crs="EPSG:4326", transform=Affine(0, 0, 0, 0, 0, 0))
    with pytest.raises(UploadError) as err:
        registry.ingest("bad.tif", bad.read_bytes())
    exc = err.value
    assert "affine" in str(exc)
    assert not list(registry.dataset_dir.glob("*.tif"))

    corrupt = tmp_path / "corrupt.tif"
    corrupt.write_bytes(b"not a tiff")
    with pytest.raises(UploadError) as err:
        registry.ingest("corrupt.tif", corrupt.read_bytes())
    exc = err.value
    assert "corrupt" in str(exc) or "unreadable" in str(exc)
    assert not list((registry.root / "tmp").glob("*"))


def test_publish_is_immutable_ordered_and_persistent(temp_service, make_tif):
    registry, _renderer, tmp = temp_service
    paths = []
    ids = []
    for idx in range(2):
        path = tmp / f"dem{idx}.tif"
        values = elevation()
        make_tif(path, values, bounds=(99.8 + idx * 0.01, 26.8, 100.0 + idx * 0.01, 27.0))
        paths.append(path)
        ids.append(registry.ingest(path.name, path.read_bytes()).id)

    published = registry.publish("v1", ids)
    assert [item.id for item in published] == ids
    with pytest.raises(VersionError) as err:
        registry.publish("v1", list(reversed(ids)))
    exc = err.value
    assert "already exists" in str(exc)

    # Same payload is idempotent, but unknown datasets are rejected.
    registry.publish("v1", ids)
    with pytest.raises(VersionError):
        registry.publish("v2", [ids[0], "missing"])

    from terrain.storage import Registry

    reloaded = Registry(root=registry.root)
    assert [item.id for item in reloaded.version("v1")] == ids
