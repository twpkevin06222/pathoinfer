"""INT8 / FP8 static post-training quantisation (PTQ) of an ONNX model folder.

Calibration tiles are read through patchology from the config's slides (``calibration``),
at the card's tile size and mpp, and preprocessed exactly as at inference
(``runtime.base.preprocess``). Two engines:

- ``modelopt`` (NVIDIA ModelOpt): Q/DQ graphs laid out for TensorRT, INT8 or FP8; nodes
  kept out of quantisation run in FP16. It runs in a child process, because ModelOpt
  patches ONNX Runtime's calibrators for the rest of the process it runs in.
- ``ort`` (ONNX Runtime): INT8 Q/DQ (symmetric, per-channel weights) that also runs in
  ONNX Runtime without TensorRT; kept nodes stay FP32.

Kept nodes come from ``keep_ops``, ``keep_nodes`` and ``keep_ends`` (``QuantConfig``).
QAT is not done here: it belongs in training (pathotrain).
"""

import multiprocessing
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger
from numpy.typing import NDArray

from pathoinfer._extras import extra, require
from pathoinfer._logging import log_call
from pathoinfer.config import QuantConfig, expand_slides
from pathoinfer.convert.common import calibration_tiles, match_nodes, write_variant
from pathoinfer.registry.loader import load_model
from pathoinfer.registry.model_card import ModelCard
from pathoinfer.runtime.base import preprocess, resolve_device

ORT_METHODS = {"entropy": "Entropy", "minmax": "MinMax", "percentile": "Percentile"}
MODELOPT_METHODS = {"entropy": "entropy", "minmax": "max", "percentile": "percentile"}


@log_call
def quantise(source: str | Path, out: Path, cfg: QuantConfig) -> ModelCard:
    """Quantise the fp32 ONNX model at ``source`` into the new folder ``out``."""
    onnx = require("onnx", extra("convert"))
    card, model_dir = load_model(str(source))
    if card.framework != "onnx" or card.precision != "fp32":
        raise ValueError(f"expected an fp32 ONNX model, got {card.framework} {card.precision}")
    cal = cfg.calibration
    tiles, tiles_hash = calibration_tiles(
        card, expand_slides(cal.slides), cal.tiles, cal.seed, cal.backend
    )
    data = preprocess(torch.from_numpy(tiles), card, resolve_device("cpu")).numpy()
    model = onnx.load(model_dir / card.file)
    for i, node in enumerate(model.graph.node):  # exclusion lists need node names
        node.name = node.name or f"{node.op_type}_{i}"
    nodes = [(n.name, n.op_type) for n in model.graph.node]
    keep = match_nodes(nodes, cfg.keep_ops, cfg.keep_nodes, cfg.keep_ends)
    logger.info("{}: keeping {} of {} nodes in high precision", card.name, len(keep), len(nodes))
    with tempfile.TemporaryDirectory() as tmp:
        named, quantised = Path(tmp) / "named.onnx", Path(tmp) / "quantised.onnx"
        onnx.save(model, named)
        if cfg.engine == "ort":
            _ort(named, quantised, card, data, keep, cfg)
        else:
            _modelopt(named, quantised, card, data, keep, cfg)
        result = onnx.load(quantised)
    settings = {
        "engine": cfg.engine,
        "method": cfg.method,
        "tiles": len(tiles),
        "kept_nodes": keep,
    }
    return write_variant(
        card,
        out,
        lambda path: onnx.save(result, path),
        precision=cfg.precision,
        version=f"{card.version}+{cfg.precision}",
        **{
            "provenance.calibration_set_hash": tiles_hash,
            "provenance.source_metadata": {
                **card.provenance.source_metadata,
                "quantisation": settings,
            },
        },
    )


def _ort(
    source: Path,
    out: Path,
    card: ModelCard,
    data: NDArray[np.float32],
    keep: list[str],
    cfg: QuantConfig,
) -> None:
    q = require("onnxruntime.quantization", extra("onnx"))

    class Reader(q.CalibrationDataReader):  # type: ignore[misc,name-defined]
        def __init__(self) -> None:
            self.batches = iter(np.array_split(data, max(1, len(data) // 16)))

        def get_next(self) -> dict[str, NDArray[Any]] | None:
            batch = next(self.batches, None)
            return None if batch is None else {card.input.name: batch}

    q.quantize_static(
        source,
        out,
        Reader(),
        quant_format=q.QuantFormat.QDQ,
        activation_type=q.QuantType.QInt8,
        weight_type=q.QuantType.QInt8,
        per_channel=True,
        nodes_to_exclude=keep,
        calibrate_method=getattr(q.CalibrationMethod, ORT_METHODS[cfg.method]),
        extra_options={
            "ActivationSymmetric": True,
            "WeightSymmetric": True,
            "CalibPercentile": cfg.percentile,
        },
    )


def _modelopt(
    source: Path,
    out: Path,
    card: ModelCard,
    data: NDArray[np.float32],
    keep: list[str],
    cfg: QuantConfig,
) -> None:
    """Run ModelOpt in a fresh process: it monkey-patches ONNX Runtime's calibrators for
    the rest of the process it runs in, which breaks later ONNX Runtime quantisation."""
    require("modelopt.onnx.quantization", extra("convert"))
    tiles = out.with_name("calibration.npy")
    np.save(tiles, data)
    with ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn")) as pool:
        pool.submit(
            _modelopt_job,
            str(source),
            str(out),
            str(tiles),
            card.input.name,
            keep,
            cfg.precision,
            MODELOPT_METHODS[cfg.method],
        ).result()


def _modelopt_job(
    source: str, out: str, tiles: str, input_name: str, keep: list[str], mode: str, method: str
) -> None:
    from modelopt.onnx.quantization import quantize

    quantize(
        onnx_path=source,
        quantize_mode=mode,
        calibration_data={input_name: np.load(tiles, mmap_mode="r")},
        calibration_method=method,
        nodes_to_exclude=[f"^{name}$" for name in keep],
        high_precision_dtype="fp16",
        output_path=out,
    )
