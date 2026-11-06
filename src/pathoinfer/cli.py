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
``pathoinfer convert fp16 SOURCE OUT``      FP32 ONNX -> FP16 ONNX model folder.
``pathoinfer convert ptq SOURCE OUT --config QUANT.yaml``
                                            INT8/FP8 static PTQ, calibrated on slides.
``pathoinfer convert trt SOURCE OUT``       TensorRT plan for this GPU (cached).
``pathoinfer convert parity REFERENCE CANDIDATE OUT --config PARITY.yaml``
                                            compare through the real stitchers; on a pass
                                            write OUT with the parity in its card (and
                                            register it with --register NAME --alias A).
``pathoinfer serve api [--jobs-dir DIR] [--max-jobs N]``
                                            the slide job API (FastAPI + uvicorn).
``pathoinfer serve triton-repo MODEL... --repository DIR``
                                            a Triton model repository from model folders.
``pathoinfer convert vlm-merge BASE ADAPTER OUT --card CARD.yaml``
                                            merge LoRA adapters into a bf16 VLM folder.
``pathoinfer convert vlm SOURCE OUT --config VLM.yaml``
                                            quantise a VLM's language model for vLLM.
``pathoinfer convert vlm-parity REFERENCE CANDIDATE OUT --config P.yaml
    --reference-url URL --candidate-url URL``   VLM parity through two vLLM servers.

``--log-level LEVEL`` (or ``-v`` for DEBUG), before the command, sets pathoinfer's and
patchology's log level; the default comes from ``PATHOINFER_LOG_LEVEL`` and
``PATCHOLOGY_LOG_LEVEL``.

Only ``run`` and ``convert`` import torch, so every other command starts fast.
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml
from loguru import logger
from pydantic import ValidationError

from patchology import SlideOpenError, open_slide
from pathoinfer._logging import PACKAGES, set_log_level
from pathoinfer.config import (
    ParityConfig,
    QuantConfig,
    RunConfig,
    VlmParityConfig,
    VlmQuantConfig,
    load_config,
    load_yaml,
)
from pathoinfer.registry.create import create_folder
from pathoinfer.registry.loader import (
    CardError,
    graph_problems,
    load_any,
    load_model,
    validate_folder,
)
from pathoinfer.registry.model_card import VqaCard
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

    convert = commands.add_parser("convert", help="convert models offline").add_subparsers(
        dest="convert_command", required=True
    )
    fp16 = convert.add_parser("fp16", help="FP32 ONNX -> FP16 ONNX")
    fp16.add_argument("source", help="model source (folder, ONNX file, models:/, hf://)")
    fp16.add_argument("out", type=Path, help="new model folder")
    fp16.add_argument(
        "--fp16-io", action="store_true", help="make the graph's inputs/outputs float16 too"
    )
    fp16.add_argument(
        "--keep-fp32", nargs="+", default=[], metavar="OP", help="op types to keep in FP32"
    )
    fp16.set_defaults(handler=_convert_fp16)
    ptq = convert.add_parser("ptq", help="INT8/FP8 static post-training quantisation")
    ptq.add_argument("source")
    ptq.add_argument("out", type=Path)
    ptq.add_argument("--config", type=Path, required=True, help="QuantConfig YAML")
    ptq.set_defaults(handler=_convert_ptq)
    plan = convert.add_parser("trt", help="TensorRT plan for this GPU")
    plan.add_argument("source")
    plan.add_argument("out", type=Path)
    plan.add_argument(
        "--batch", type=int, nargs=3, default=[1, 32, 64], metavar=("MIN", "OPT", "MAX")
    )
    plan.add_argument("--workspace-gb", type=float, default=4.0)
    plan.add_argument("--device", type=int, default=0, help="CUDA device index")
    plan.set_defaults(handler=_convert_trt)
    parity = convert.add_parser("parity", help="parity gate for a converted model")
    parity.add_argument("reference")
    parity.add_argument("candidate")
    parity.add_argument("out", type=Path, help="new model folder written on a pass")
    parity.add_argument("--config", type=Path, required=True, help="ParityConfig YAML")
    parity.add_argument("--register", metavar="NAME", help="MLflow registered model name")
    parity.add_argument("--alias", help="MLflow alias, e.g. fp16-blackwell (with --register)")
    parity.add_argument("--tracking-uri", help="MLflow tracking URI (default: environment)")
    parity.add_argument("--report", type=Path, help="write the parity report as JSON")
    parity.set_defaults(handler=_convert_parity)
    serve = commands.add_parser("serve", help="serve models and slide jobs").add_subparsers(
        dest="serve_command", required=True
    )
    api = serve.add_parser("api", help="run the slide job API")
    api.add_argument("--host", default="0.0.0.0")
    api.add_argument("--port", type=int, default=8080)
    api.add_argument("--jobs-dir", type=Path, default=Path("jobs"))
    api.add_argument("--max-jobs", type=int, default=1, help="jobs running at once")
    api.set_defaults(handler=_serve_api)
    repo = serve.add_parser("triton-repo", help="write a Triton model repository")
    repo.add_argument("models", nargs="+", help="model sources (folders, models:/, hf://)")
    repo.add_argument("--repository", type=Path, required=True)
    repo.add_argument("--config", type=Path, help="TritonRepoConfig YAML")
    repo.set_defaults(handler=_serve_triton_repo)
    merge = convert.add_parser("vlm-merge", help="merge LoRA adapters into a VLM")
    merge.add_argument("base", help="base model folder or Hugging Face Hub id")
    merge.add_argument("adapter", help="PEFT adapter folder or Hub id")
    merge.add_argument("out", type=Path)
    merge.add_argument("--card", type=Path, required=True, help="VqaCard fields (YAML)")
    merge.set_defaults(handler=_convert_vlm_merge)
    vlm = convert.add_parser("vlm", help="quantise a VLM's language model (llm-compressor)")
    vlm.add_argument("source", help="bf16 VLM folder with a VqaCard")
    vlm.add_argument("out", type=Path)
    vlm.add_argument("--config", type=Path, required=True, help="VlmQuantConfig YAML")
    vlm.set_defaults(handler=_convert_vlm)
    vparity = convert.add_parser("vlm-parity", help="VLM parity gate through vLLM servers")
    vparity.add_argument("reference")
    vparity.add_argument("candidate")
    vparity.add_argument("out", type=Path, help="new folder written on a pass")
    vparity.add_argument("--config", type=Path, required=True, help="VlmParityConfig YAML")
    vparity.add_argument("--reference-url", required=True, help="vLLM serving the reference")
    vparity.add_argument("--candidate-url", required=True, help="vLLM serving the candidate")
    vparity.add_argument("--register", metavar="NAME")
    vparity.add_argument("--alias")
    vparity.add_argument("--tracking-uri")
    vparity.add_argument("--report", type=Path)
    vparity.set_defaults(handler=_convert_vlm_parity)
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


def _convert_fp16(args: argparse.Namespace) -> int:
    from pathoinfer.convert.fp16 import to_fp16

    to_fp16(args.source, args.out, keep_io_types=not args.fp16_io, keep_fp32_ops=args.keep_fp32)
    return 0


def _convert_ptq(args: argparse.Namespace) -> int:
    from pathoinfer.convert.ptq import quantise

    quantise(args.source, args.out, load_yaml(args.config, QuantConfig))
    return 0


def _convert_trt(args: argparse.Namespace) -> int:
    from pathoinfer.convert.tensorrt import build_plan

    low, opt, high = args.batch
    build_plan(
        args.source,
        args.out,
        min_batch=low,
        opt_batch=opt,
        max_batch=high,
        workspace_gb=args.workspace_gb,
        device=args.device,
    )
    return 0


def _convert_parity(args: argparse.Namespace) -> int:
    from pathoinfer.convert.parity import gate

    if bool(args.register) != bool(args.alias):
        raise ValueError("--register and --alias go together")
    register = (args.register, args.alias) if args.register else None
    report = gate(
        args.reference,
        args.candidate,
        load_yaml(args.config, ParityConfig),
        args.out,
        register,
        args.tracking_uri,
    )
    if args.report:
        args.report.write_text(json.dumps(asdict(report), indent=2) + "\n")
    return 0 if report.passed else 1


def _serve_api(args: argparse.Namespace) -> int:
    from pathoinfer._extras import extra, require
    from pathoinfer.serving.api import create_app

    uvicorn = require("uvicorn", extra("serve"))
    uvicorn.run(create_app(args.jobs_dir, args.max_jobs), host=args.host, port=args.port)
    return 0


def _serve_triton_repo(args: argparse.Namespace) -> int:
    from pathoinfer.serving.triton_repo import TritonRepoConfig, write_repository

    cfg = load_yaml(args.config, TritonRepoConfig) if args.config else TritonRepoConfig()
    write_repository(args.models, args.repository, cfg)
    return 0


def _convert_vlm_merge(args: argparse.Namespace) -> int:
    from pathoinfer.convert.vlm import merge_lora

    merge_lora(args.base, args.adapter, args.out, yaml.safe_load(args.card.read_text()))
    return 0


def _convert_vlm(args: argparse.Namespace) -> int:
    from pathoinfer.convert.vlm import quantise_vlm

    quantise_vlm(args.source, args.out, load_yaml(args.config, VlmQuantConfig))
    return 0


def _convert_vlm_parity(args: argparse.Namespace) -> int:
    from pathoinfer.convert.vlm_parity import gate, report_json

    if bool(args.register) != bool(args.alias):
        raise ValueError("--register and --alias go together")
    report = gate(
        args.reference,
        args.candidate,
        load_yaml(args.config, VlmParityConfig),
        args.out,
        args.reference_url,
        args.candidate_url,
        (args.register, args.alias) if args.register else None,
        args.tracking_uri,
    )
    if args.report:
        args.report.write_text(report_json(report))
    return 0 if report.passed else 1


def _info_model(args: argparse.Namespace) -> int:
    any_card, model_dir = load_any(args.source)
    if isinstance(any_card, VqaCard):
        logger.info(
            "{} {} | vqa | {} {} | {} px at {} mpp | grounding {} | folder {}",
            any_card.name,
            any_card.version,
            any_card.architecture,
            any_card.precision,
            any_card.input.tile_size,
            any_card.input.mpp,
            any_card.output.grounding,
            model_dir,
        )
        return 0
    card = any_card
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
