"""Compare a converted model with its full-precision reference through the real stitchers.

Both models run on the same tiles: mini-slides, i.e. windows of ``window_tiles`` x
``window_tiles`` grid tiles around tissue on each listed slide, read once through
patchology. Each window's outputs go through the task's stitcher (cores, seams, polygons)
for both models, and the stitched results are compared per class, pooled over windows:

- segmentation: Dice of the class's polygons (``2|A n B| / (|A| + |B|)``);
- detection: F1 of detections matched one to one by box IoU >= ``iou``, and the
  relative count difference ``|n_candidate - n_reference| / max(n_reference, 1)``;
- instance: the same two metrics for nuclei, matched one to one by polygon IoU >= ``iou``
  (background is never written, so it is not scored);
- classification: one-vs-rest AUC of the candidate's core probabilities against the
  reference's argmax labels.

The worst class of each metric is checked against its tolerance. A pass writes the
candidate as a new model folder whose card records the parity (``provenance.parity``);
a fail lists the failing metrics, the worst classes and, for two ONNX models, the layers
whose outputs diverge most, and writes nothing.
"""

import hashlib
import json
import os
import shutil
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import shapely
import torch
from loguru import logger
from numpy.typing import NDArray

from patchology import open_slide
from pathoinfer._extras import extra, require
from pathoinfer._logging import log_call
from pathoinfer.config import ParityConfig, RuntimeConfig, expand_slides
from pathoinfer.convert.common import CARD_FILE
from pathoinfer.registry.loader import load_model
from pathoinfer.registry.model_card import (
    ClassificationCard,
    DetectionCard,
    InstanceCard,
    ModelCard,
    Parity,
    SegmentationCard,
    parse_any_card,
    parse_card,
)
from pathoinfer.runtime.base import Runtime, preprocess, resolve_device
from pathoinfer.runtime.factory import make_runtime
from pathoinfer.stitch.base import Stitcher, StitchResult
from pathoinfer.stitch.detection import DetectionStitcher
from pathoinfer.stitch.heatmap import HeatmapStitcher
from pathoinfer.stitch.instance import InstanceStitcher
from pathoinfer.stitch.segmentation import SegmentationStitcher
from pathoinfer.tiling.grid import SlideGrid, build_grid, read_tile

Window = tuple[SlideGrid, NDArray[np.uint8]]  # a mini-slide grid and its tiles, NHWC


@dataclass(frozen=True)
class ParityReport:
    passed: bool
    metrics: dict[str, float]  # worst class per metric
    tolerances: dict[str, float]
    per_class: dict[str, dict[str, float]]  # metric -> class name -> value
    failures: list[str]
    tile_set_hash: str
    tiles: int
    worst_layers: list[tuple[str, float]] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"parity {'PASS' if self.passed else 'FAIL'} on {self.tiles} tiles"]
        for name, value in self.metrics.items():
            worst = min(self.per_class[name].items(), key=lambda kv: _badness(name, kv[1]))
            lines.append(
                f"  {name} {value:.4f} (bound {self.tolerances[name]:g}; worst class {worst[0]})"
            )
        lines += [f"  failed: {f}" for f in self.failures]
        lines += [f"  layer {n}: relative error {e:.3g}" for n, e in self.worst_layers]
        return "\n".join(lines)


def windows(card: ModelCard, cfg: ParityConfig) -> Iterator[Window]:
    """Mini-slides: per slide, ``cfg.windows`` windows of up to ``window_tiles``² kept
    tiles, each centred on a random tissue tile (seeded)."""
    rng = np.random.default_rng(cfg.seed)
    overlap = round(card.recommended_overlap * card.input.tile_size)
    for path in expand_slides(cfg.slides):
        with open_slide(path, cfg.backend) as slide:
            grid = build_grid(
                slide, card.input.tile_size, card.input.mpp, overlap, icc=card.input.icc
            )
            if not grid.tiles:
                logger.warning("{}: no tissue tiles; skipped", path.name)
                continue
            half = cfg.window_tiles // 2
            for centre in rng.choice(len(grid.tiles), size=cfg.windows, replace=False):
                r0 = grid.tiles[centre].row - half
                c0 = grid.tiles[centre].col - half
                tiles = tuple(
                    s
                    for s in grid.tiles
                    if r0 <= s.row < r0 + cfg.window_tiles and c0 <= s.col < c0 + cfg.window_tiles
                )
                mini = replace(grid, tiles=tiles)
                yield mini, np.stack([read_tile(slide, mini, s) for s in tiles])


def tile_set_hash(sets: list[Window]) -> str:
    keys = [(g.slide_id, s.x, s.y, g.tile_size, g.mpp) for g, _ in sets for s in g.tiles]
    return hashlib.sha256(json.dumps(keys).encode()).hexdigest()[:16]


def compare(
    card: ModelCard,
    reference: Runtime,
    candidate: Runtime,
    sets: list[Window],
    cfg: ParityConfig,
) -> tuple[dict[str, dict[str, float]], int]:
    """Per-class metrics of ``candidate`` against ``reference`` on ``sets``, and the
    number of tiles."""
    pooled: dict[str, Any] = {}
    tiles = 0
    for grid, images in sets:
        results = []
        for runtime in (reference, candidate):
            stitcher = _stitcher(card, grid, cfg)
            batch = runtime.batch_size if hasattr(runtime, "batch_size") else 16
            for start in range(0, len(images), batch):
                specs = list(grid.tiles[start : start + batch])
                stitcher.add(
                    specs, runtime.predict(torch.from_numpy(images[start : start + batch]))
                )
            results.append(stitcher.finalize())
        _accumulate(card, pooled, results[0], results[1], cfg.iou)
        tiles += len(images)
    return _finish(card, pooled), tiles


def check(reference: str | Path, candidate: str | Path, cfg: ParityConfig) -> ParityReport:
    """Run both model sources on ``cfg``'s mini-slides and judge the candidate."""
    ref_card, ref_dir = load_model(str(reference))
    cand_card, cand_dir = load_model(str(candidate))
    _same_contract(ref_card, cand_card)
    ref_runtime = make_runtime(
        ref_card,
        ref_dir,
        RuntimeConfig(
            device=cfg.reference_device, batch_size=cfg.candidate.batch_size, tensorrt=False
        ),
    )
    cand_runtime = make_runtime(cand_card, cand_dir, cfg.candidate)
    sets = list(windows(ref_card, cfg))
    if not sets:
        raise ValueError("no tissue tiles in the parity slides")
    per_class, tiles = compare(ref_card, ref_runtime, cand_runtime, sets, cfg)
    tolerances = cfg.tolerances.model_dump()
    metrics = {name: _worst(name, values) for name, values in per_class.items()}
    bounds = {name: tolerances[name] for name in metrics}
    failures = [
        f"{name} {value:.4f} {'>' if name.endswith('_max') else '<'} {bounds[name]:g}"
        for name, value in metrics.items()
        if _badness(name, value) > _badness(name, bounds[name])
    ]
    worst_layers: list[tuple[str, float]] = []
    if failures and ref_card.framework == cand_card.framework == "onnx":
        images = np.concatenate([imgs for _, imgs in sets])[:8]
        worst_layers = layer_errors(ref_card, ref_dir, cand_dir / cand_card.file, images)
    report = ParityReport(
        not failures, metrics, bounds, per_class, failures, tile_set_hash(sets), tiles, worst_layers
    )
    logger.info("{}", report.summary())
    return report


@log_call
def gate(
    reference: str | Path,
    candidate: str | Path,
    cfg: ParityConfig,
    out: Path,
    register: tuple[str, str] | None = None,
    tracking_uri: str | None = None,
) -> ParityReport:
    """``check``; on a pass, copy the candidate folder to the new folder ``out`` with
    ``provenance.parity`` in its card and, with ``register=(name, alias)``, register it in
    MLflow under that alias. On a fail, write nothing."""
    report = check(reference, candidate, cfg)
    if not report.passed:
        return report
    if out.exists():
        raise FileExistsError(f"{out} exists; choose a new folder")
    ref_card, _ = load_model(str(reference))
    cand_card, cand_dir = load_model(str(candidate))
    data = cand_card.model_dump(mode="json")
    data["provenance"]["parity"] = Parity(
        reference=f"{ref_card.name}@{ref_card.version}",
        tile_set_hash=report.tile_set_hash,
        metrics=report.metrics,
        tolerances=report.tolerances,
        checked_at=datetime.now(UTC).isoformat(timespec="seconds"),
    ).model_dump(mode="json")
    card = parse_card(data)
    partial = out.with_name(out.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    shutil.copytree(cand_dir, partial)
    (partial / CARD_FILE).write_text(card.model_dump_json(indent=2) + "\n")
    os.replace(partial, out)
    logger.info("parity passed: wrote {}", out)
    if register is not None:
        register_model(out, *register, tracking_uri=tracking_uri)
    return report


def register_model(folder: Path, name: str, alias: str, tracking_uri: str | None = None) -> str:
    """Log ``folder`` as an MLflow run's artifacts, add it as a new version of registered
    model ``name`` and point ``alias`` at it; returns the version."""
    mlflow = require("mlflow", extra("mlflow"))
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    card = parse_any_card(json.loads((folder / CARD_FILE).read_text()))
    experiment = client.get_experiment_by_name("pathoinfer-convert") or client.get_experiment(
        client.create_experiment("pathoinfer-convert")
    )
    run = client.create_run(experiment.experiment_id, run_name=f"{card.name}-{card.version}")
    client.log_artifacts(run.info.run_id, str(folder), "model")
    client.set_terminated(run.info.run_id)
    if not client.search_registered_models(filter_string=f"name = '{name}'"):
        client.create_registered_model(name)
    version = client.create_model_version(
        name, f"{run.info.artifact_uri}/model", run_id=run.info.run_id
    )
    client.set_registered_model_alias(name, alias, version.version)
    logger.info("registered {} version {} as @{}", name, version.version, alias)
    return str(version.version)


def layer_errors(
    card: ModelCard,
    ref_dir: Path,
    candidate: Path,
    images: NDArray[np.uint8],
    top: int = 5,
    budget_bytes: int = 1 << 30,
) -> list[tuple[str, float]]:
    """The ``top`` intermediate tensors (by name, present in both ONNX graphs) whose
    candidate values differ most from the reference's: relative L2 error on ``images``.

    Tensors are fetched in groups of about ``budget_bytes`` per graph (sizes from the
    reference graph's inferred shapes; a tensor of unknown size runs alone), one session run
    per group, so memory stays bounded: all of HoVer-Net's tensors at once took 60 GB."""
    onnx = require("onnx", extra("convert"))
    ort = require("onnxruntime", extra("onnx"))
    graphs = [onnx.load(ref_dir / card.file), onnx.load(candidate)]
    names = [{o for n in g.graph.node for o in n.output if o} for g in graphs]
    shared = sorted(names[0] & names[1])
    sizes = _tensor_bytes(onnx, graphs[0], len(images))
    x = preprocess(torch.from_numpy(images), card, resolve_device("cpu"))
    errors = []
    for group in _groups(shared, sizes, budget_bytes):
        r_values, c_values = (_run_outputs(onnx, ort, g, group, x) for g in graphs)
        for n in group:
            r, c = r_values[n], c_values[n]
            if r.shape != c.shape or not np.issubdtype(r.dtype, np.floating):
                continue
            r64, c64 = r.astype(np.float64), c.astype(np.float64)
            errors.append((n, float(np.linalg.norm(c64 - r64) / max(np.linalg.norm(r64), 1e-12))))
    return sorted(errors, key=lambda e: -e[1])[:top]


def _tensor_bytes(onnx: Any, model: Any, batch: int) -> dict[str, int]:
    """Bytes of each node output whose shape inference resolves, for ``batch`` inputs."""
    sized = onnx.ModelProto()
    sized.CopyFrom(model)
    for dim in sized.graph.input[0].type.tensor_type.shape.dim[:1]:
        if not dim.dim_value:
            dim.dim_value = batch
    try:
        inferred = onnx.shape_inference.infer_shapes(sized)
    except (onnx.shape_inference.InferenceError, ValueError):  # >2 GB models: ValueError
        return {}
    out = {}
    for v in [*inferred.graph.value_info, *inferred.graph.output]:
        t = v.type.tensor_type
        dims = [d.dim_value for d in t.shape.dim]
        if t.elem_type and dims and all(dims):
            itemsize = onnx.helper.tensor_dtype_to_np_dtype(t.elem_type).itemsize
            out[v.name] = int(np.prod(dims)) * itemsize
    return out


def _groups(names: list[str], sizes: dict[str, int], budget: int) -> Iterator[list[str]]:
    """``names`` in order, packed into groups whose known sizes sum to at most ``budget``."""
    group: list[str] = []
    used = 0
    for n in names:
        size = sizes.get(n, budget)
        if group and used + size > budget:
            yield group
            group, used = [], 0
        group.append(n)
        used += size
    if group:
        yield group


def _run_outputs(
    onnx: Any, ort: Any, model: Any, names: list[str], x: torch.Tensor
) -> dict[str, NDArray[Any]]:
    """``model`` run on CPU with exactly ``names`` as its outputs."""
    probe = onnx.ModelProto()
    probe.CopyFrom(model)
    del probe.graph.output[:]
    probe.graph.output.extend(onnx.helper.make_empty_tensor_value_info(n) for n in names)
    options = ort.SessionOptions()
    options.enable_cpu_mem_arena = False  # an arena per session keeps its peak until exit
    session = ort.InferenceSession(
        probe.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    feed = {session.get_inputs()[0].name: _as_input(x, session)}
    return dict(zip(names, session.run(names, feed), strict=True))


def _as_input(x: torch.Tensor, session: Any) -> NDArray[Any]:
    dtype = np.float16 if "float16" in session.get_inputs()[0].type else np.float32
    return x.cpu().numpy().astype(dtype)


def _same_contract(reference: ModelCard, candidate: ModelCard) -> None:
    if reference.task != candidate.task or [c.name for c in reference.classes] != [
        c.name for c in candidate.classes
    ]:
        raise ValueError("reference and candidate differ in task or classes")
    if (reference.input.tile_size, reference.input.mpp) != (
        candidate.input.tile_size,
        candidate.input.mpp,
    ):
        raise ValueError("reference and candidate differ in tile size or mpp")


def _stitcher(card: ModelCard, grid: SlideGrid, cfg: ParityConfig) -> Stitcher:
    if isinstance(card, SegmentationCard):
        seg = cfg.segmentation
        return SegmentationStitcher(
            card,
            grid,
            output_mpp=seg.output_mpp or max(2.0, grid.mpp),
            simplify_px=seg.simplify_px,
            smooth=seg.smooth,
            min_area_um2=seg.min_area_um2,
            fill_holes_um2=seg.fill_holes_um2,
            connectivity=seg.connectivity,
        )
    if isinstance(card, DetectionCard):
        det = cfg.detection
        return DetectionStitcher(
            card, grid, score_threshold=det.score_threshold, nms_iou=det.nms_iou
        )
    if isinstance(card, InstanceCard):
        return InstanceStitcher(card, grid, **cfg.instance.model_dump())
    if not isinstance(card, ClassificationCard):
        raise NotImplementedError(f"parity for {card.task} models is not supported yet")
    return HeatmapStitcher(card, grid)


def _accumulate(
    card: ModelCard, pooled: dict[str, Any], ref: StitchResult, cand: StitchResult, iou: float
) -> None:
    for c in card.classes:
        acc = pooled.setdefault(
            c.name,
            {
                "inter": 0.0,
                "ref": 0.0,
                "cand": 0.0,
                "tp": 0,
                "nr": 0,
                "nc": 0,
                "scores": [],
                "labels": [],
            },
        )
        if isinstance(card, SegmentationCard):
            a = shapely.union_all([f.geometry for f in ref.features if f.class_id == c.id])
            b = shapely.union_all([f.geometry for f in cand.features if f.class_id == c.id])
            acc["inter"] += shapely.intersection(a, b).area
            acc["ref"] += a.area
            acc["cand"] += b.area
        elif isinstance(card, (DetectionCard, InstanceCard)):
            r, k = _class_shapes(ref, c.id), _class_shapes(cand, c.id)
            acc["tp"] += _matches(r, k, iou)
            acc["nr"] += len(r)
            acc["nc"] += len(k)
        else:
            ref_probs = ref.detections.class_probs
            cand_probs = cand.detections.class_probs
            acc["labels"].append(ref_probs.argmax(1) == c.id)
            acc["scores"].append(cand_probs[:, c.id])


def _finish(card: ModelCard, pooled: dict[str, Any]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    background = getattr(card, "background_id", None)
    for c in card.classes:
        acc = pooled[c.name]
        if c.id == background and not isinstance(card, ClassificationCard):
            continue
        if isinstance(card, SegmentationCard):
            total = acc["ref"] + acc["cand"]
            out.setdefault("dice_min", {})[c.name] = 1.0 if total == 0 else 2 * acc["inter"] / total
        elif isinstance(card, (DetectionCard, InstanceCard)):
            denominator = acc["nr"] + acc["nc"]
            out.setdefault("f1_min", {})[c.name] = (
                1.0 if denominator == 0 else 2 * acc["tp"] / denominator
            )
            out.setdefault("count_diff_max", {})[c.name] = abs(acc["nc"] - acc["nr"]) / max(
                acc["nr"], 1
            )
        else:
            labels = np.concatenate(acc["labels"])
            scores = np.concatenate(acc["scores"])
            out.setdefault("auc_min", {})[c.name] = auc(labels, scores)
    return out


def _class_shapes(result: StitchResult, class_id: int) -> NDArray[np.object_]:
    """A class's objects as shapely geometries: detection boxes, or instance polygons."""
    if result.instances is not None:
        found = result.instances
        return np.asarray(found.geometry[found.class_id == class_id])
    d = result.detections
    return np.asarray(shapely.box(*np.asarray(d.boxes[d.labels == class_id]).T))


def _matches(ref: NDArray[np.object_], cand: NDArray[np.object_], iou: float) -> int:
    """One-to-one matches by IoU >= ``iou``, greedy from the highest IoU."""
    if len(ref) == 0 or len(cand) == 0:
        return 0
    i, j = shapely.STRtree(cand).query(ref, predicate="intersects")
    inter = shapely.area(shapely.intersection(ref[i], cand[j]))
    overlap = inter / (shapely.area(ref[i]) + shapely.area(cand[j]) - inter)
    used_r: set[int] = set()
    used_c: set[int] = set()
    for k in np.argsort(-overlap):
        if overlap[k] < iou:
            break
        if i[k] not in used_r and j[k] not in used_c:
            used_r.add(int(i[k]))
            used_c.add(int(j[k]))
    return len(used_r)


def auc(labels: NDArray[np.bool_], scores: NDArray[np.floating]) -> float:
    """Area under the ROC curve (Mann-Whitney, ties counted half); 1.0 when one of the two
    groups is empty, as nothing can be ranked wrongly."""
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return 1.0
    ranks = _ranks(np.concatenate([pos, neg]))
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _ranks(values: NDArray[np.floating]) -> NDArray[np.float64]:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values))
    ranks[order] = np.arange(1, len(values) + 1)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    sums = np.bincount(inverse, weights=ranks)
    result: NDArray[np.float64] = (sums / counts)[inverse]
    return result


def _worst(name: str, values: dict[str, float]) -> float:
    return max(values.values()) if name.endswith("_max") else min(values.values())


def _badness(name: str, value: float) -> float:
    return value if name.endswith("_max") else -value
