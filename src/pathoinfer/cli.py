"""Command line interface.

``pathoinfer run CONFIG``                   run a config's slides; exit 1 if any failed.
``pathoinfer validate CONFIG``              check a config, its model and its slides
                                            without running anything.
``pathoinfer info model SOURCE``            show a model's card (folder, models:/, hf://).
``pathoinfer info slide PATH``              show a slide's size, levels and resolution.
``pathoinfer card create MODEL OUT``        wrap an ONNX / PyTorch / TensorRT / safetensors
                                            file in a model folder; fields come from the
                                            file, a legacy ``pathotrain`` key, ``--fields``
                                            and the flags.
``pathoinfer card validate FOLDER``         check a model folder's card and, for ONNX, that
                                            the graph matches it.

``--log-level LEVEL`` (or ``-v`` for DEBUG), before the command, sets pathoinfer's and
patchology's log level; the default comes from ``PATHOINFER_LOG_LEVEL`` and
``PATCHOLOGY_LOG_LEVEL``.

Only ``run`` imports torch, so every other command starts fast.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml
from loguru import logger
from pydantic import ValidationError

from patchology import SlideOpenError, open_slide
from pathoinfer._logging import PACKAGES, set_log_level
from pathoinfer.config import RunConfig, load_config
from pathoinfer.registry.create import create_folder
from pathoinfer.registry.loader import CardError, graph_problems, load_model, validate_folder
from pathoinfer.tiling.slide_id import slide_id

LEVELS = ["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL", "ALL"]


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.log_level:
        for package in PACKAGES:
            set_log_level(args.log_level, package)
    try:
        return int(args.handler(args))
    except (
        CardError,
        ValidationError,
        ValueError,
        FileNotFoundError,
        SlideOpenError,
        NotImplementedError,
    ) as e:
        logger.error("{}", _describe(e))
        return 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pathoinfer")
    level = parser.add_mutually_exclusive_group()
    level.add_argument(
        "--log-level",
        type=str.upper,
        choices=LEVELS,
        help="log level for pathoinfer and patchology (default: their *_LOG_LEVEL, else INFO)",
    )
    level.add_argument(
        "-v", "--verbose", action="store_const", const="DEBUG", dest="log_level", help="DEBUG"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run a config over its slides")
    run.add_argument("config", type=Path)
    run.set_defaults(handler=_run)
    validate = commands.add_parser("validate", help="check a config without running it")
    validate.add_argument("config", type=Path)
    validate.set_defaults(handler=_validate_config)

    info = commands.add_parser("info", help="describe a model or a slide").add_subparsers(
        dest="info_command", required=True
    )
    model = info.add_parser("model", help="show a model's card")
    model.add_argument(
        "source", help="model folder, ONNX file, models:/NAME@ALIAS or hf://ORG/REPO"
    )
    model.set_defaults(handler=_info_model)
    slide = info.add_parser("slide", help="show a slide's size, levels and resolution")
    slide.add_argument("path", type=Path)
    slide.add_argument("--backend", default="openslide", choices=["openslide", "fastslide"])
    slide.set_defaults(handler=_info_slide)

    card = commands.add_parser("card", help="create or validate model folders").add_subparsers(
        dest="card_command", required=True
    )
    create = card.add_parser("create", help="wrap a model file in a model folder")
    create.add_argument("model", type=Path, help="model file")
    create.add_argument("out", type=Path, help="new model folder")
    create.add_argument("--fields", type=Path, help="YAML or JSON with (partial) card fields")
    create.add_argument("--name", help="card name (default: the file stem)")
    create.add_argument("--version", help="card version")
    create.set_defaults(handler=_create)
    check = card.add_parser("validate", help="validate a model folder")
    check.add_argument("folder", type=Path)
    check.set_defaults(handler=lambda args: _validate_card(args.folder))
    return parser


def _run(args: argparse.Namespace) -> int:
    from pathoinfer.engine.pipeline import run_batch  # imports torch

    results = run_batch(load_config(args.config))
    return 1 if any(r.status == "failed" for r in results) else 0


def _validate_config(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    problems = _config_problems(cfg)
    for problem in problems:
        logger.error("{}", problem)
    if not problems:
        logger.info("ok: {} ({}, {} slides)", args.config, cfg.task, len(cfg.slide_paths()))
    return 1 if problems else 0


def _config_problems(cfg: RunConfig) -> list[str]:
    problems = []
    slides = cfg.slide_paths()
    if not slides:
        problems.append(f"no slides match {cfg.slides!r}")
    problems += [f"slide not found: {p}" for p in slides if not p.exists()]
    try:
        card, _ = load_model(cfg.model)
    except (CardError, FileNotFoundError, ImportError) as e:
        return [*problems, f"model {cfg.model}: {e}"]
    if card.task != cfg.task:
        problems.append(f"config task is {cfg.task}, but {cfg.model} is a {card.task} model")
    if card.task == "classification":
        problems.append("classification runs land in v0.2")
    unknown = set(cfg.class_colors) - {c.name for c in card.classes}
    if unknown:
        problems.append(f"class_colors names unknown classes {sorted(unknown)}")
    output_mpp = cfg.segmentation.output_mpp
    if card.task == "segmentation" and output_mpp is not None and output_mpp < card.input.mpp:
        problems.append(f"segmentation.output_mpp {output_mpp} is finer than the model's")
    return problems


def _info_model(args: argparse.Namespace) -> int:
    card, model_dir = load_model(args.source)
    logger.info(
        "{} {} | {} | {} {} ({}) | {} px at {} mpp, {} {} {} | overlap {:.2f} | folder {}",
        card.name,
        card.version,
        card.task,
        card.framework,
        card.file,
        card.precision,
        card.input.tile_size,
        card.input.mpp,
        card.input.channel_order,
        card.input.layout,
        card.input.dtype,
        card.recommended_overlap,
        model_dir,
    )
    background = getattr(card, "background_id", None)
    for c in card.classes:
        logger.info(
            "class {} {} {}{}", c.id, c.name, list(c.color), " (background)" * (c.id == background)
        )
    logger.info("output {}", card.output.model_dump_json())
    return 0


def _info_slide(args: argparse.Namespace) -> int:
    with open_slide(args.path, args.backend) as slide:
        mpp = slide.mpp
        logger.info(
            "{} | {} x {} px | mpp {} | slide_id {}",
            args.path.name,
            *slide.dimensions,
            "missing (inference will refuse this slide)"
            if mpp is None
            else f"{mpp[0]:.4g} x {mpp[1]:.4g}",
            slide_id(args.path),
        )
        for level, ((w, h), d) in enumerate(
            zip(slide.level_dimensions, slide.level_downsamples, strict=True)
        ):
            logger.info("level {}: {} x {} px, downsample {:.4g}", level, w, h, d)
    return 0


def _create(args: argparse.Namespace) -> int:
    fields: dict[str, Any] = yaml.safe_load(args.fields.read_text()) if args.fields else {}
    fields |= {k: v for k in ("name", "version") if (v := getattr(args, k)) is not None}
    card = create_folder(args.model, args.out, fields)
    logger.info("wrote {} ({}, {} {})", args.out, card.task, card.name, card.version)
    return 0


def _validate_card(folder: Path) -> int:
    card = validate_folder(folder)
    try:
        problems = graph_problems(card, folder)
    except ImportError as e:
        logger.warning("graph not checked: {}", e)
        problems = []
    for problem in problems:
        logger.error("{}", problem)
    if not problems:
        logger.info("ok: {} ({}, {} {})", folder, card.task, card.name, card.version)
    return 1 if problems else 0


def _describe(error: Exception) -> str:
    if isinstance(error, ValidationError):
        return "invalid config:\n" + "\n".join(
            f"  {'.'.join(map(str, e['loc']))}: {e['msg']}" for e in error.errors()
        )
    return str(error)


if __name__ == "__main__":
    sys.exit(main())
