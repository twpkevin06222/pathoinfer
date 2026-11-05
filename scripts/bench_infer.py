"""Inference throughput (tiles/s) per runtime and precision for model folders::

    uv run python scripts/bench_infer.py MODEL_DIR [MODEL_DIR ...] [--batch-size 16]

Each model folder is benchmarked with every runtime its framework allows: ONNX models on
CPU and CUDA (TensorRT EP too with --tensorrt), PyTorch models on CPU and CUDA with no
autocast, bf16 and fp16. Input is a pinned uint8 batch, as the DataLoader delivers it, so
the timing includes the host-to-device copy and preprocessing. Each configuration warms
up, then runs ``--iters`` batches ``--repeats`` times; the table shows the median and
range. Outputs are checked finite, and the right shape, every batch.
"""

import argparse
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from pathoinfer.config import RuntimeConfig
from pathoinfer.registry.loader import load_model
from pathoinfer.runtime.factory import make_runtime


def configs(framework: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    devices = ["cpu", "cuda"] if torch.cuda.is_available() and not args.cpu_only else ["cpu"]
    if framework == "onnx":
        return [{"device": d, "tensorrt": args.tensorrt and d == "cuda"} for d in devices]
    autocasts: list[str | None] = [None, "bf16", "fp16"]
    return [
        {"device": d, "autocast": a} for d in devices for a in autocasts if d == "cuda" or a is None
    ]


def bench(folder: Path, config: dict[str, Any], args: argparse.Namespace) -> list[float]:
    card, model_dir = load_model(folder)
    runtime = make_runtime(card, model_dir, RuntimeConfig(batch_size=args.batch_size, **config))
    size = card.input.tile_size
    batch = torch.randint(0, 256, (args.batch_size, size, size, 3), dtype=torch.uint8)
    if runtime.device.type == "cuda":
        batch = batch.pin_memory()
    for _ in range(args.warmup):
        runtime.predict(batch)
    iters = args.iters if runtime.device.type == "cuda" else max(1, args.iters // 10)
    rates = []
    for _ in range(args.repeats):
        _sync(runtime.device)
        start = time.perf_counter()
        for _ in range(iters):
            outputs = runtime.predict(batch)
            for name, out in outputs.items():
                assert out.shape[0] == args.batch_size, (name, out.shape)
        _sync(runtime.device)
        rates.append(iters * args.batch_size / (time.perf_counter() - start))
        for out in outputs.values():
            assert torch.isfinite(out.float()).all(), "non-finite output"
    return rates


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("models", type=Path, nargs="+", help="model folders")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--iters", type=int, default=30, help="batches per repeat (CPU: 1/10)")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tensorrt", action="store_true", help="let ONNX models use TensorRT")
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    print(f"GPU: {gpu}; batch {args.batch_size}; median (min-max) of {args.repeats} repeats")
    print(f"{'model':<24} {'runtime':<8} {'device':<6} {'precision':<12} {'tiles/s':>20}")
    for folder in args.models:
        card, _ = load_model(folder)
        for config in configs(card.framework, args):
            rates = bench(folder, config, args)
            mode = config.get("autocast") or ("trt" if config.get("tensorrt") else None)
            precision = card.precision + (f"+{mode}" if mode else "")
            spread = f"{statistics.median(rates):.0f} ({min(rates):.0f}-{max(rates):.0f})"
            print(
                f"{card.name:<24} {card.framework:<8} {config['device']:<6} {precision:<12} {spread:>20}"
            )


if __name__ == "__main__":
    main()
