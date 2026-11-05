"""Command line interface. Lands in v0.1.

``pathoinfer run CONFIG SLIDE...``  run the engine over slides.
``pathoinfer card create MODEL``    wrap any ONNX / PyTorch model (third-party, or a legacy
                                    pathotrain ONNX, prefilled from its ``pathotrain`` key)
                                    in a model folder with a ``model_card.json``.
``pathoinfer card validate PATH``   check a card against ``model_card.v1.json``.
"""

import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="pathoinfer")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run a config over slides")
    run.add_argument("config")
    run.add_argument("slides", nargs="+")
    card = commands.add_parser("card", help="create or validate model cards").add_subparsers(
        dest="card_command", required=True
    )
    card.add_parser("create", help="wrap a model file in a model folder").add_argument("model")
    card.add_parser("validate", help="validate a model card").add_argument("path")
    args = parser.parse_args(argv)
    name = args.command if args.command == "run" else f"card {args.card_command}"
    raise NotImplementedError(f"pathoinfer {name} lands in v0.1")
