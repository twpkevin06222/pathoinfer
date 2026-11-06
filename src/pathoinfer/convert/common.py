"""Shared by the converters: write a variant model folder, sample calibration tiles, pick
the layers to keep in high precision."""

import fnmatch
import hashlib
import json
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from patchology import open_slide
from pathoinfer.registry.model_card import ModelCard, parse_card
from pathoinfer.tiling.grid import build_grid, read_tile

CARD_FILE = "model_card.json"


def write_variant(
    card: ModelCard, out: Path, write_model: Callable[[Path], object], **updates: Any
) -> ModelCard:
    """Write ``out/<file>`` with ``write_model`` and a card derived from ``card``:
    ``updates`` (dotted keys such as ``input.dtype``) applied and ``parent_model`` set.
    ``out`` must not exist; it appears complete or not at all."""
    if out.exists():
        raise FileExistsError(f"{out} exists; choose a new folder")
    data = card.model_dump(mode="json")
    data["provenance"] = data["provenance"] | {
        "parent_model": f"{card.name}@{card.version}",
        "parity": None,
    }
    for key, value in updates.items():
        target = data
        *path, leaf = key.split(".")
        for part in path:
            target = target[part]
        target[leaf] = value
    variant = parse_card(data)
    partial = out.with_name(out.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    write_model(partial / variant.file)
    (partial / CARD_FILE).write_text(variant.model_dump_json(indent=2) + "\n")
    os.replace(partial, out)
    logger.info("wrote {} ({} {}, {})", out, variant.name, variant.version, variant.precision)
    return variant


def calibration_tiles(
    card: ModelCard, slides: list[Path], n_tiles: int, seed: int = 0, backend: str = "openslide"
) -> tuple[NDArray[np.uint8], str]:
    """``n_tiles`` uint8 NHWC tiles at the card's tile size and mpp, spread evenly over
    ``slides`` (tissue tiles only, chosen at random with ``seed``), and a hash of exactly
    which tiles were read (slide ids, positions, tiling)."""
    if not slides:
        raise ValueError("no calibration slides")
    rng = np.random.default_rng(seed)
    per_slide = np.full(len(slides), n_tiles // len(slides))
    per_slide[: n_tiles % len(slides)] += 1
    tiles, keys = [], []
    for path, count in zip(slides, per_slide, strict=True):
        with open_slide(path, backend) as slide:  # type: ignore[arg-type]
            grid = build_grid(slide, card.input.tile_size, card.input.mpp, icc=card.input.icc)
            if not grid.tiles:
                logger.warning("{}: no tissue tiles; skipped for calibration", path.name)
                continue
            picks = rng.choice(len(grid.tiles), size=min(count, len(grid.tiles)), replace=False)
            for i in sorted(picks):
                spec = grid.tiles[i]
                tiles.append(read_tile(slide, grid, spec))
                keys.append((grid.slide_id, spec.x, spec.y))
    if not tiles:
        raise ValueError("no tissue tiles in the calibration slides")
    blob = json.dumps(
        {"tiles": keys, "tile_size": card.input.tile_size, "mpp": card.input.mpp}
    ).encode()
    logger.info("calibration: {} tiles from {} slides", len(tiles), len(slides))
    return np.stack(tiles), hashlib.sha256(blob).hexdigest()[:16]


def match_nodes(
    nodes: list[tuple[str, str]], op_types: list[str], name_patterns: list[str], ends: bool
) -> list[str]:
    """Names of graph nodes (``(name, op_type)`` in topological order) to keep in high
    precision: those whose op type is in ``op_types``, whose name matches a glob in
    ``name_patterns``, and with ``ends`` the first and last compute node (Conv, MatMul,
    Gemm)."""
    keep = {n for n, op in nodes if op in op_types}
    keep |= {n for n, _ in nodes for p in name_patterns if fnmatch.fnmatch(n.lower(), p)}
    compute = [n for n, op in nodes if op in ("Conv", "ConvTranspose", "MatMul", "Gemm")]
    if ends and compute:
        keep |= {compute[0], compute[-1]}
    return [n for n, _ in nodes if n in keep]
