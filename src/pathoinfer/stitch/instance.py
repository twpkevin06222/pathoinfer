"""Instance stitcher: HoVer-Net style tile outputs to slide-level nuclei, each reported once.

Per tile (``post/instances.py``): the ``np`` softmax gives each pixel's nucleus
probability, HoVer-Net's watershed on it and the ``hv`` maps gives an instance map, and
each instance takes the majority non-background type of the ``nt`` argmax under it.
Instances without a type are dropped (background is never written).

Ownership, as for detections: an instance is kept only by the tile whose core contains its
centroid. Cores partition the slide and lie ``overlap // 2`` model pixels inside their
tile, so a nucleus up to about ``overlap`` wide is seen whole by its owner and reported
exactly once; a tile that sees it cut at its edge has its (shifted) centroid outside its
core and drops it. Larger objects can be reported twice or cut; a warning is logged when
the overlap is below the card's ``recommended_overlap``.

Polygons (``post/vectorise.py``): a tile's kept instances are traced together from its
label map, so touching nuclei share one border line; areas equal pixel counts before
smoothing. Each instance's ``score`` is the mean probability of its type over its pixels;
``measurements`` add ``perimeter_um`` and ``nucleus_probability`` (the mean ``np``
probability). Both probabilities are usable as uncertainty for review.

Everything is held as NumPy columns (one chunk per batch), read by ``io.objects.ObjectTable``.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from loguru import logger
from numpy.typing import NDArray

from pathoinfer.post.instances import instance_types, softmax, watershed_instances
from pathoinfer.post.vectorise import label_pieces, to_features
from pathoinfer.registry.model_card import InstanceCard
from pathoinfer.runtime.base import Outputs
from pathoinfer.stitch.base import StitchResult
from pathoinfer.tiling.grid import SlideGrid, TileSpec


@dataclass(frozen=True)
class Instances:
    """Slide-level instances as columns, level-0 pixels: one row per nucleus."""

    geometry: NDArray[np.object_]
    class_id: NDArray[np.int64]
    score: NDArray[np.float32]
    area_um2: NDArray[np.float64]
    measurements: dict[str, NDArray[np.float64]]

    def __len__(self) -> int:
        return len(self.class_id)


MEASUREMENTS = ("perimeter_um", "nucleus_probability")


class InstanceStitcher:
    def __init__(
        self,
        card: InstanceCard,
        grid: SlideGrid,
        *,
        threshold: float = 0.5,
        marker_threshold: float = 0.4,
        min_size_px: int = 10,
        simplify_px: float = 1.0,
        smooth: int = 2,
    ) -> None:
        self.card, self.grid = card, grid
        self.threshold, self.marker_threshold, self.min_size = (
            threshold,
            marker_threshold,
            min_size_px,
        )
        self.simplify_px, self.smooth = simplify_px, smooth
        self.names = {c.id: c.name for c in card.classes}
        recommended = round(card.recommended_overlap * grid.tile_size)
        if grid.overlap < recommended:
            logger.warning(
                "grid overlap {} px is below the card's recommended {} px; nuclei wider than "
                "about the overlap may be reported twice or cut",
                grid.overlap,
                recommended,
            )
        self._columns: list[dict[str, Any]] = []

    def add(self, specs: list[TileSpec], outputs: Outputs) -> None:
        out = self.card.output
        nuclei = softmax(_numpy(outputs[out.np]), axis=1)[:, 1]
        hv, types = _numpy(outputs[out.hv]), _numpy(outputs[out.nt])
        for i, spec in enumerate(specs):
            self._tile(spec, nuclei[i], hv[i], types[i])

    def _tile(
        self,
        spec: TileSpec,
        nuclei: NDArray[np.float32],
        hv: NDArray[np.float32],
        type_logits: NDArray[np.float32],
    ) -> None:
        inst = watershed_instances(nuclei, hv, self.threshold, self.marker_threshold, self.min_size)
        if not inst.any():
            return
        grid = self.grid
        kinds = instance_types(inst, type_logits)
        n = len(kinds)
        sizes = np.bincount(inst.ravel(), minlength=n)
        ys, xs = np.indices(inst.shape)
        # Centroids in level-0 pixels (pixel centres), to test against the tile's core.
        cx = np.bincount(inst.ravel(), weights=xs.ravel() + 0.5, minlength=n)
        cy = np.bincount(inst.ravel(), weights=ys.ravel() + 0.5, minlength=n)
        with np.errstate(invalid="ignore", divide="ignore"):
            cx = (spec.xm + cx / sizes) * grid.downsample
            cy = (spec.ym + cy / sizes) * grid.downsample
        core = spec.core
        owned = (cx >= core.x0) & (cx < core.x1) & (cy >= core.y0) & (cy < core.y1)
        keep = owned & (kinds > 0) & (sizes > 0)
        keep[0] = False
        if not keep.any():
            return
        inst = np.where(keep[inst], inst, 0)
        probs = softmax(type_logits)
        type_prob = np.take_along_axis(probs, kinds[inst][None], 0)[0]
        nucleus_sum = np.bincount(inst.ravel(), weights=nuclei.ravel(), minlength=n)
        pieces = label_pieces(inst, type_prob, kinds)  # one per kept id, in increasing id order
        for piece, i in zip(pieces, np.flatnonzero(keep), strict=True):
            piece.extra = {"nucleus_probability": float(nucleus_sum[i] / sizes[i])}
        features = to_features(
            pieces,
            self.names,
            level0_per_px=grid.downsample,
            slide_mpp=grid.slide_mpp,
            origin=(spec.xm * grid.downsample, spec.ym * grid.downsample),
            simplify_px=self.simplify_px,
            smooth=self.smooth,
        )
        if not features:
            return
        self._columns.append(
            {
                "geometry": [f.geometry for f in features],
                "class_id": [f.class_id for f in features],
                "score": [f.measurements["mean_probability"] for f in features],
                "area_um2": [f.measurements["area_um2"] for f in features],
                **{k: [f.measurements[k] for f in features] for k in MEASUREMENTS},
            }
        )

    def finalize(self) -> StitchResult:
        c = self._columns
        keys = ("geometry", "class_id", "score", "area_um2", *MEASUREMENTS)
        joined = {k: [v for chunk in c for v in chunk[k]] for k in keys}
        geometry = np.empty(len(joined["geometry"]), dtype=object)
        geometry[:] = joined["geometry"]
        instances = Instances(
            geometry=geometry,
            class_id=np.array(joined["class_id"], np.int64),
            score=np.array(joined["score"], np.float32),
            area_um2=np.array(joined["area_um2"], np.float64),
            measurements={k: np.array(joined[k], np.float64) for k in MEASUREMENTS},
        )
        logger.info("instance: {} objects", len(instances))
        return StitchResult(features=[], instances=instances)


def _numpy(tensor: torch.Tensor) -> NDArray[Any]:
    return tensor.detach().float().cpu().numpy()
