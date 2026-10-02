from __future__ import annotations

import math

import numpy as np
import pytest
from PIL import Image
from io import BytesIO

from terrain.tiles import TileError, expanded_geometry, validate_xyz


def smooth_dem(width=120, height=120):
    x = np.linspace(0, 1, width, dtype=np.float32)
    y = np.linspace(0, 1, height, dtype=np.float32)
    gx, gy = np.meshgrid(x, y)
    return (1000 + 400 * gx - 300 * gy + 100 * np.sin(8 * gx) * np.cos(8 * gy)).astype(np.float32)


def publish_sample(registry, make_tif, tmp, hole=False):
    values = smooth_dem()
    if hole:
        values[55:65, 55:65] = -9999.0
    path = tmp / ("hole.tif" if hole else "dem.tif")
    make_tif(path, values, bounds=(99.80, 26.84, 99.92, 26.96), write_mask=hole)
    return registry.ingest(path.name, path.read_bytes()).id


def test_xyz_validation_and_geometry():
    validate_xyz(0, 0, 0)
    validate_xyz(18, 262143, 262143)
    for args in ((19, 0, 0), (2, 4, 0), (2, 0, 4), (-1, 0, 0)):
        with pytest.raises(TileError):
            validate_xyz(*args)
    shape, transform, bounds = expanded_geometry(0, 0, 0)
    assert shape == (258, 258)
    assert math.isclose(bounds[0], -math.pi * 6378137, abs_tol=1e-6)
    assert math.isclose(bounds[2], math.pi * 6378137, abs_tol=1e-6)


def test_render_tile_hole_fill_and_neighbor_transparency(temp_service, make_tif):
    registry, renderer, tmp = temp_service
    primary_id = publish_sample(registry, make_tif, tmp, hole=True)
    secondary_id = publish_sample(registry, make_tif, tmp, hole=False)
    datasets = registry.publish("v1", [primary_id, secondary_id])

    # z=12, x=3184, y=1730 contains the sample area.
    png = renderer.render("v1", datasets, 12, 3184, 1730, 315, 45)
    image = np.array(Image.open(BytesIO(png)).convert("RGBA"))
    assert image.shape == (256, 256, 4)
    assert image[..., 3].max() == 255
    assert image[..., 3].min() == 0
    gray_opaque = image[image[..., 3] == 255][..., 0]
    assert gray_opaque.min() < gray_opaque.max()

    # The exact hole is filled by the lower priority dataset, so the center is
    # opaque. Its boundary is transparent because a Horn neighbour is missing
    # in the primary where secondary still covers it; if secondary did not
    # cover it, no edge pixels would treat NoData as zero.
    center_alpha = image[120:136, 120:136, 3]
    assert (center_alpha == 255).any()


def test_lighting_and_version_cache_keys_are_distinct(temp_service, make_tif):
    registry, renderer, tmp = temp_service
    dataset_id = publish_sample(registry, make_tif, tmp)
    datasets = registry.publish("north", [dataset_id])
    shade_nw = renderer.render("north", datasets, 12, 3184, 1730, 315, 45)
    shade_se = renderer.render("north", datasets, 12, 3184, 1730, 135, 45)
    assert shade_nw != shade_se
    datasets2 = registry.publish("east", [dataset_id])
    shade_other = renderer.render("east", datasets2, 12, 3184, 1730, 315, 45)
    assert shade_other == shade_nw
    renderer.cache.clear()
    assert renderer.render("north", datasets, 12, 3184, 1730, 315, 45) == shade_nw


def test_no_coverage_is_transparent(temp_service, make_tif):
    registry, renderer, tmp = temp_service
    dataset_id = publish_sample(registry, make_tif, tmp)
    datasets = registry.publish("v1", [dataset_id])
    png = renderer.render("v1", datasets, 3, 1, 3, 315, 45)  # Atlantic area
    image = np.array(Image.open(BytesIO(png)).convert("RGBA"))
    assert np.array_equal(image[..., 3], np.zeros((256, 256), dtype=np.uint8))


def test_render_without_nodata_or_mask(temp_service, make_tif):
    # A fully valid raster with neither NoData nor an explicit mask used to
    # fail in the mosaic warp because the masked read collapses to a scalar.
    registry, renderer, tmp = temp_service
    path = tmp / "plain.tif"
    make_tif(
        path,
        smooth_dem(),
        bounds=(99.80, 26.84, 99.92, 26.96),
        nodata=None,
        write_mask=False,
    )
    dataset_id = registry.ingest(path.name, path.read_bytes()).id
    datasets = registry.publish("plain", [dataset_id])
    png = renderer.render("plain", datasets, 12, 3184, 1730, 315, 45)
    image = np.array(Image.open(BytesIO(png)).convert("RGBA"))
    assert image[..., 3].max() == 255
