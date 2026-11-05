"""Process slides: tiling -> parallel reads -> batching -> runtime -> stitcher thread ->
writers. Lands in v0.1.
"""

from pathlib import Path

from pathoinfer.config import InferenceConfig


def run_slide(slide: Path, config: InferenceConfig, out_dir: Path) -> list[Path]:
    raise NotImplementedError("lands in v0.1")


def run_batch(slides: list[Path], config: InferenceConfig, out_dir: Path) -> dict[Path, list[Path]]:
    raise NotImplementedError("lands in v0.1")
