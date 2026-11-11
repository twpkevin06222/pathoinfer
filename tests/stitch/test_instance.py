import numpy as np
import pytest
import shapely
import torch

from pathoinfer.registry.model_card import InstanceCard, parse_card
from pathoinfer.stitch.instance import InstanceStitcher
from pathoinfer.tiling.grid import SlideGrid
from tests.post.test_instances import ideal_hv

from .test_segmentation import CORNER, DOWNSAMPLE, TILE, grid

# Nuclei in level-0 pixels: (x, y, radius, type). One sits on the corner where four tiles
# meet, one on a vertical seam only, the rest inside single tiles; type 0 has no type.
CORNER_PX = CORNER
NUCLEI = [
    (CORNER_PX, CORNER_PX, 24, 1),
    ((4 * (TILE - 32) + 16) * DOWNSAMPLE, 1000.0, 20, 2),
    (300.0, 300.0, 22, 1),
    (2000.0, 600.0, 18, 2),
    (900.0, 2500.0, 16, 0),
]


def card(overlap: float = 32 / TILE) -> InstanceCard:
    parsed = parse_card(
        {
            "name": "nuclei",
            "version": "1",
            "task": "instance",
            "framework": "onnx",
            "file": "model.onnx",
            "input": {
                "tile_size": TILE,
                "mpp": 0.5,
                "normalisation": {"mean": [0.5] * 3, "std": [0.5] * 3},
            },
            "classes": [
                {"id": 0, "name": "background", "color": [0, 0, 0]},
                {"id": 1, "name": "tumour", "color": [255, 0, 0]},
                {"id": 2, "name": "lymphocyte", "color": [0, 0, 255]},
            ],
            "output": {"n_classes": 3},
            "recommended_overlap": overlap,
        }
    )
    assert isinstance(parsed, InstanceCard)
    return parsed


def nucleus_map(g: SlideGrid) -> tuple[np.ndarray, np.ndarray]:
    """The slide's nuclei at model resolution: an instance map and each id's type."""
    side = g.n_cols * (TILE - g.overlap) + g.overlap
    inst = np.zeros((side, side), np.int32)
    yy, xx = np.mgrid[:side, :side] + 0.5
    for i, (x, y, r, _) in enumerate(NUCLEI, start=1):
        cx, cy, rr = x / DOWNSAMPLE, y / DOWNSAMPLE, r / DOWNSAMPLE
        inst[(xx - cx) ** 2 + (yy - cy) ** 2 < rr**2] = i
    return inst, np.array([0, *(t for *_, t in NUCLEI)])


def ideal_outputs(g: SlideGrid) -> dict[str, torch.Tensor]:
    """What a perfect model returns per tile: the nuclei it sees, cut at its edges."""
    inst, types = nucleus_map(g)
    nps, hvs, nts = [], [], []
    for s in g.tiles:
        crop = inst[s.ym : s.ym + TILE, s.xm : s.xm + TILE]
        nucleus = (crop > 0).astype(np.float32)
        nps.append(np.stack([10 - 20 * nucleus, 20 * nucleus - 10]))  # confident either way
        hvs.append(ideal_hv(crop))
        nt = np.zeros((3, TILE, TILE), np.float32)
        for k in range(3):
            nt[k][types[crop] == k] = 10.0
        nts.append(nt)
    return {
        name: torch.from_numpy(np.stack(v)) for name, v in (("np", nps), ("hv", hvs), ("nt", nts))
    }


def stitch(c: InstanceCard, g: SlideGrid) -> tuple[np.ndarray, ...]:
    stitcher = InstanceStitcher(c, g)
    stitcher.add(list(g.tiles), ideal_outputs(g))
    found = stitcher.finalize().instances
    centres = shapely.get_coordinates(shapely.centroid(found.geometry))
    return found, centres


def test_every_typed_nucleus_is_reported_once_with_its_type() -> None:
    g = grid()
    found, centres = stitch(card(), g)
    typed = [(x, y, r, t) for x, y, r, t in NUCLEI if t > 0]
    assert len(found) == len(typed)  # the untyped nucleus is not written
    for x, y, r, t in typed:
        hits = np.flatnonzero(np.hypot(*(centres - [x, y]).T) < r / 2)
        assert len(hits) == 1, f"nucleus at {(x, y)} found {len(hits)} times"
        (k,) = hits
        assert found.class_id[k] == t
        assert found.area_um2[k] == pytest.approx(np.pi * r**2 * g.slide_mpp**2, rel=0.1)
    assert (found.score > 0.99).all() and (found.measurements["nucleus_probability"] > 0.99).all()


def test_nuclei_on_seams_and_corners_are_owned_by_one_tile() -> None:
    g = grid()
    owners = []
    for x, y, *_ in NUCLEI[:2]:  # the corner and the seam nucleus
        owners.append(
            [s for s in g.tiles if s.core.x0 <= x < s.core.x1 and s.core.y0 <= y < s.core.y1]
        )
        seeing = [
            s
            for s in g.tiles
            if s.xm * DOWNSAMPLE <= x < (s.xm + TILE) * DOWNSAMPLE
            and s.ym * DOWNSAMPLE <= y < (s.ym + TILE) * DOWNSAMPLE
        ]
        assert len(seeing) > 1  # really on a seam: several tiles see it
    assert all(len(o) == 1 for o in owners)
    found, _ = stitch(card(), g)
    assert len(found) == sum(t > 0 for *_, t in NUCLEI)


def test_low_overlap_warns(caplog: pytest.LogCaptureFixture) -> None:
    InstanceStitcher(card(overlap=0.5), grid())
    assert "below the card's recommended" in caplog.text


def test_no_nuclei_gives_an_empty_table() -> None:
    g = grid()
    empty = {
        "np": torch.zeros(len(g.tiles), 2, TILE, TILE),
        "hv": torch.zeros(len(g.tiles), 2, TILE, TILE),
        "nt": torch.zeros(len(g.tiles), 3, TILE, TILE),
    }
    stitcher = InstanceStitcher(card(), g)
    stitcher.add(list(g.tiles), empty)
    found = stitcher.finalize().instances
    assert len(found) == 0 and found.geometry.dtype == object
