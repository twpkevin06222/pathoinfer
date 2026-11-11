import numpy as np
import shapely
from scipy import ndimage as ndi

from pathoinfer.post.instances import instance_types, softmax, watershed_instances
from pathoinfer.post.vectorise import label_pieces


def touching_nuclei() -> np.ndarray:
    """Three nuclei on a 64 px tile, two of them touching along x = 26."""
    inst = np.zeros((64, 64), np.int32)
    yy, xx = np.mgrid[:64, :64]
    inst[(yy - 20) ** 2 + (xx - 18) ** 2 < 81] = 1
    inst[(yy - 20) ** 2 + (xx - 34) ** 2 < 81] = 2
    inst[(yy - 46) ** 2 + (xx - 40) ** 2 < 64] = 3
    return inst


def ideal_hv(inst: np.ndarray) -> np.ndarray:
    """A perfect model's HV maps: each pixel's offset from its nucleus's centre, scaled to
    [-1, 1] on each side (the training target)."""
    hv = np.zeros((2, *inst.shape), np.float32)
    for i, box in enumerate(ndi.find_objects(inst), start=1):
        if box is None:  # an id absent from this map (e.g. a tile's crop)
            continue
        inside = inst[box] == i
        cy, cx = ndi.center_of_mass(inside)
        ys, xs = np.nonzero(inside)
        for channel, offset in ((0, xs - cx), (1, ys - cy)):
            offset = offset.astype(np.float32)
            offset[offset < 0] /= -offset[offset < 0].min()
            offset[offset > 0] /= offset[offset > 0].max()
            hv[channel][box][inside] = offset
    return hv


def test_watershed_separates_touching_nuclei_from_ideal_outputs() -> None:
    inst = touching_nuclei()
    pred = watershed_instances((inst > 0).astype(np.float32), ideal_hv(inst))
    assert pred.max() == 3
    for i in (1, 2, 3):  # each true nucleus is one predicted instance, nearly exactly
        ids, counts = np.unique(pred[inst == i], return_counts=True)
        assert counts.max() / counts.sum() > 0.9 and ids[counts.argmax()] > 0


def test_nothing_above_threshold_gives_no_instances() -> None:
    assert watershed_instances(np.zeros((32, 32), np.float32), np.zeros((2, 32, 32))).max() == 0
    small = np.zeros((32, 32), np.float32)
    small[4:7, 4:7] = 1.0  # 9 px, under min_size
    assert watershed_instances(small, np.zeros((2, 32, 32))).max() == 0


def test_types_are_the_majority_non_background_argmax() -> None:
    inst = touching_nuclei()
    logits = np.zeros((3, 64, 64), np.float32)
    logits[1][inst == 1] = 5.0  # nucleus 1: type 1
    logits[2][inst == 2] = 5.0  # nucleus 2: type 2
    logits[0][inst == 3] = 5.0  # nucleus 3: background everywhere -> no type
    assert instance_types(inst, logits).tolist() == [0, 1, 2, 0]
    probs = softmax(logits)
    assert np.allclose(probs.sum(0), 1) and probs.dtype == np.float32


def test_label_pieces_keep_touching_instances_apart_with_exact_areas() -> None:
    inst = touching_nuclei()
    pieces = label_pieces(inst, np.ones(inst.shape, np.float32), np.array([0, 1, 2, 1]))
    assert [p.class_id for p in pieces] == [1, 2, 1]
    for i, piece in enumerate(pieces, start=1):
        assert piece.geometry.area == piece.pixels == (inst == i).sum()
    shared = shapely.intersection(pieces[0].geometry, pieces[1].geometry)
    assert shared.area == 0 and shared.length > 0  # one border, no overlap
    assert label_pieces(np.zeros((4, 4), np.int32), np.zeros((4, 4), np.float32), np.zeros(1)) == []
