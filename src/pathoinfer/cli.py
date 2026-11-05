"""Command line interface.

``pathoinfer run CONFIG SLIDE...``          run the engine over slides (lands in v0.1).
``pathoinfer card create MODEL OUT``        wrap an ONNX / PyTorch / TensorRT / safetensors
                                            file in a model folder; fields come from the
                                            file, a legacy ``pathotrain`` key, ``--fields``
                                            and the flags.
``pathoinfer card validate FOLDER``         check a model folder's card and, for ONNX, that
                                            the graph matches it.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

from pathoinfer.registry.create import create_folder
from pathoinfer.registry.loader import CardError, graph_problems, validate_folder


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pathoinfer")
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

    if args.command == "run":
        raise NotImplementedError("pathoinfer run lands in v0.1")
    try:
        if args.card_command == "create":
            return _create(args)
        return _validate(args.folder)
    except CardError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


def _create(args: argparse.Namespace) -> int:
    fields: dict[str, Any] = yaml.safe_load(args.fields.read_text()) if args.fields else {}
    fields |= {k: v for k in ("name", "version") if (v := getattr(args, k)) is not None}
    card = create_folder(args.model, args.out, fields)
    print(f"wrote {args.out} ({card.task}, {card.name} {card.version})")
    return 0


def _validate(folder: Path) -> int:
    card = validate_folder(folder)
    try:
        problems = graph_problems(card, folder)
    except ImportError as e:
        print(f"warning: graph not checked: {e}", file=sys.stderr)
        problems = []
    for problem in problems:
        print(f"error: {problem}", file=sys.stderr)
    if not problems:
        print(f"ok: {folder} ({card.task}, {card.name} {card.version})")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
