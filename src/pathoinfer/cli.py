"""Command line interface.

``pathoinfer run CONFIG SLIDE...``          run the engine over slides (lands in v0.1).
``pathoinfer card create MODEL OUT``        wrap an ONNX / PyTorch / TensorRT / safetensors
                                            file in a model folder; fields come from the
                                            file, a legacy ``pathotrain`` key, ``--fields``
                                            and the flags.
``pathoinfer card validate FOLDER``         check a model folder's card and, for ONNX, that
                                            the graph matches it.

``--log-level LEVEL`` (or ``-v`` for DEBUG), before the command, sets pathoinfer's and
patchology's log level; the default comes from ``PATHOINFER_LOG_LEVEL`` and
``PATCHOLOGY_LOG_LEVEL``.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from pathoinfer._logging import PACKAGES, set_log_level
from pathoinfer.registry.create import create_folder
from pathoinfer.registry.loader import CardError, graph_problems, validate_folder


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pathoinfer")
    level = parser.add_mutually_exclusive_group()
    level.add_argument(
        "--log-level",
        type=str.upper,
        choices=["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL", "ALL"],
        help="log level for pathoinfer and patchology (default: their *_LOG_LEVEL, else INFO)",
    )
    level.add_argument(
        "-v", "--verbose", action="store_const", const="DEBUG", dest="log_level", help="DEBUG"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run a config over slides")
    run.add_argument("config")
    run.add_argument("slides", nargs="+")
    card = commands.add_parser("card", help="create or validate model folders").add_subparsers(
        dest="card_command", required=True
    )
    create = card.add_parser("create", help="wrap a model file in a model folder")
    create.add_argument("model", type=Path, help="model file")
    create.add_argument("out", type=Path, help="new model folder")
    create.add_argument("--fields", type=Path, help="YAML or JSON with (partial) card fields")
    create.add_argument("--name", help="card name (default: the file stem)")
    create.add_argument("--version", help="card version")
    validate = card.add_parser("validate", help="validate a model folder")
    validate.add_argument("folder", type=Path)
    args = parser.parse_args(argv)
    if args.log_level:
        for package in PACKAGES:
            set_log_level(args.log_level, package)

    if args.command == "run":
        raise NotImplementedError("pathoinfer run lands in v0.1")
    try:
        if args.card_command == "create":
            return _create(args)
        return _validate(args.folder)
    except CardError as e:
        logger.error("{}", e)
        return 1


def _create(args: argparse.Namespace) -> int:
    fields: dict[str, Any] = yaml.safe_load(args.fields.read_text()) if args.fields else {}
    fields |= {k: v for k in ("name", "version") if (v := getattr(args, k)) is not None}
    card = create_folder(args.model, args.out, fields)
    logger.info("wrote {} ({}, {} {})", args.out, card.task, card.name, card.version)
    return 0


def _validate(folder: Path) -> int:
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


if __name__ == "__main__":
    sys.exit(main())
