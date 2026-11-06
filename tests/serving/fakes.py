"""Fake engines for the job API tests. Module-level, so spawned workers can import them."""

import time
from pathlib import Path

from pathoinfer.serving.api import JobRequest, Progress


def quick(request: JobRequest, folder: Path, progress: Progress) -> list[Path]:
    out = folder / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    for done in (2, 4):
        progress(done, 4)
    (out / "slide.geojson").write_text('{"type": "FeatureCollection", "features": []}')
    (out / "nested").mkdir()
    (out / "nested" / "summary.parquet").write_bytes(b"PAR1")
    return [out / "slide.geojson", out / "nested" / "summary.parquet"]


def slow(request: JobRequest, folder: Path, progress: Progress) -> list[Path]:
    """Runs until stopped; reports progress so the test can see it running."""
    (folder / "started").touch()
    for done in range(1, 10_000):
        progress(done, 10_000)
        time.sleep(0.05)
    return []


def failing(request: JobRequest, folder: Path, progress: Progress) -> list[Path]:
    raise RuntimeError("the model exploded")
