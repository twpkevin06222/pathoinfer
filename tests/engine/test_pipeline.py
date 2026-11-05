"""End to end: a synthetic pyramidal slide through the fixture models, every output read
back. Model cards run at 4 mpp (16 px tiles of 256 level-0 px), so a slide is ~100 tiles."""

import gzip
import json
import multiprocessing
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
import shapely
import tifffile
import torch
import yaml

from pathoinfer.cli import main
from pathoinfer.config import RunConfig
from pathoinfer.engine.pipeline import load, run_batch, run_slide

FIXTURES = Path(__file__).parents[1] / "fixtures"
MPP = 4.0


def model(tmp_path: Path, fixture: str) -> Path:
    folder = shutil.copytree(FIXTURES / fixture, tmp_path / fixture)
    card = json.loads((folder / "model_card.json").read_text())
    card["input"]["mpp"] = MPP
    (folder / "model_card.json").write_text(json.dumps(card))
    return folder


def config(tmp_path: Path, slides: Path, task: str, /, **sections: Any) -> RunConfig:
    """``task`` picks the fixture model; ``sections`` override the config's keys."""
    fixture = {"segmentation": "tiny_seg", "detection": "tiny_det"}[task]
    data: dict[str, Any] = {
        "task": task,
        "model": str(model(tmp_path, fixture)),
        "slides": str(slides / "slide_name-01.tiff"),
        "runtime": {"device": "cpu", "batch_size": 16},
        "loader": {"num_workers": 0},
        "outputs": {"dir": str(tmp_path / "out"), "tiff": True},
    }
    for key, value in sections.items():
        data[key] = data.get(key, {}) | value if isinstance(value, dict) else value
    return RunConfig.model_validate(data)


def features(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt") as f:
        data: dict[str, Any] = json.load(f)
    return data["features"]


def test_segmentation_writes_every_output(tmp_path: Path, slides: Path) -> None:
    cfg = config(tmp_path, slides, "segmentation")
    (result,) = run_batch(cfg)
    assert result.status == "done" and result.tiles > 50
    out = tmp_path / "out"
    names = {p.name for p in out.iterdir()}
    assert {
        "slide_name-01.geojson.gz",
        "slide_name-01.objects.parquet",
        "slide_name-01.summary.parquet",
        "slide_name-01.labels.tiff",
        "slide_name-01.done.json",
    } <= names
    assert any(n.startswith("batch-") for n in names) and any(n.startswith("run-") for n in names)
    geometries = [
        shapely.from_geojson(json.dumps(f["geometry"]))
        for f in features(out / "slide_name-01.geojson.gz")
    ]
    assert geometries and all(g.is_valid for g in geometries)
    objects = pl.read_parquet(out / "slide_name-01.objects.parquet")
    assert len(objects) == len(geometries) == result.objects
    assert (objects["slide_id"] == result.slide_id).all()
    with tifffile.TiffFile(out / "slide_name-01.labels.tiff") as tif:
        labels = tif.series[0].levels[0].asarray()
    assert set(np.unique(labels)) <= {-1, 0, 1}


def test_detection_reports_one_object_per_tile(tmp_path: Path, slides: Path) -> None:
    cfg = config(
        tmp_path, slides, "detection", outputs={"per_class": True}, detection={"geometry": "point"}
    )
    (result,) = run_batch(cfg)
    assert result.status == "done"
    assert result.objects == result.tiles
    objects = pl.read_parquet(tmp_path / "out" / "slide_name-01.objects.parquet")
    assert len(objects) == result.tiles
    np.testing.assert_allclose(objects["score"].to_numpy(), 0.9)
    per_class = sorted((tmp_path / "out").glob("slide_name-01.*.geojson.gz"))
    assert sum(len(features(p)) for p in per_class) == result.tiles
    assert all(f["geometry"]["type"] == "Point" for p in per_class for f in features(p))


def test_resume_skips_finished_slides_until_settings_change(tmp_path: Path, slides: Path) -> None:
    cfg = config(tmp_path, slides, "detection")
    assert [r.status for r in run_batch(cfg)] == ["done"]
    assert [r.status for r in run_batch(cfg)] == ["skipped"]
    changed = cfg.model_copy(
        update={"detection": cfg.detection.model_copy(update={"score_threshold": 0.8})}
    )
    assert [r.status for r in run_batch(changed)] == ["done"]
    (tmp_path / "out" / "slide_name-01.objects.parquet").unlink()
    assert [r.status for r in run_batch(changed)] == ["done"]
    no_resume = changed.model_copy(update={"resume": False})
    assert [r.status for r in run_batch(no_resume)] == ["done"]


def test_failed_slides_are_recorded_and_the_batch_continues(tmp_path: Path, slides: Path) -> None:
    cfg = config(
        tmp_path,
        slides,
        "detection",
        slides=[
            str(slides / "slide_name-02.tiff"),
            str(slides / "missing.svs"),
            str(slides / "slide_name-01.tiff"),
        ],
    )
    results = run_batch(cfg)
    assert [r.status for r in results] == ["failed", "failed", "done"]
    assert "SlideResolutionError" in (results[0].error or "")
    (summary,) = (tmp_path / "out").glob("batch-*.parquet")
    frame = pl.read_parquet(summary)
    assert frame["status"].to_list() == ["failed", "failed", "done"]
    assert frame["error"].null_count() == 1


class Exploding:
    """A runtime that fails on its third batch."""

    def __init__(self, inner: Any) -> None:
        self.inner, self.calls = inner, 0

    def predict(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        self.calls += 1
        if self.calls == 3:
            raise RuntimeError("boom")
        out: dict[str, torch.Tensor] = self.inner.predict(batch)
        return out

    def warmup(self) -> None:
        pass


def test_an_error_mid_slide_stops_the_reader_processes(tmp_path: Path, slides: Path) -> None:
    cfg = config(tmp_path, slides, "segmentation", loader={"num_workers": 2})
    loaded = load(cfg)
    broken = type(loaded)(
        loaded.card, loaded.model_dir, Exploding(loaded.runtime), loaded.settings_hash
    )  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="boom"):
        run_slide(cfg, slides / "slide_name-01.tiff", broken)
    assert multiprocessing.active_children() == []
    assert not (tmp_path / "out" / "slide_name-01.done.json").exists()


def test_reader_processes_give_the_same_result(tmp_path: Path, slides: Path) -> None:
    serial = run_slide(config(tmp_path, slides, "detection"), slides / "slide_name-01.tiff")
    parallel = run_slide(
        config(tmp_path / "p", slides, "detection", loader={"num_workers": 2}),
        slides / "slide_name-01.tiff",
    )
    assert (serial.tiles, serial.objects) == (parallel.tiles, parallel.objects)
    assert multiprocessing.active_children() == []


def test_cli_validates_and_runs_a_config(tmp_path: Path, slides: Path) -> None:
    cfg = config(tmp_path, slides, "segmentation")
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(cfg.model_dump(mode="json")))
    assert main(["validate", str(path)]) == 0
    assert main(["info", "model", cfg.model]) == 0
    assert main(["info", "slide", str(slides / "slide_name-01.tiff")]) == 0
    assert main(["run", str(path)]) == 0
    assert (tmp_path / "out" / "slide_name-01.done.json").is_file()


def test_cli_reports_bad_configs(
    tmp_path: Path, slides: Path, caplog: pytest.LogCaptureFixture
) -> None:
    cfg = config(
        tmp_path, slides, "segmentation", task="detection", slides=str(slides / "nope.svs")
    )
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(cfg.model_dump(mode="json")))
    assert main(["validate", str(path)]) == 1
    assert "slide not found" in caplog.text and "is a segmentation model" in caplog.text
    path.write_text("task: segmentation\nmodel: m\nslides: s\nloader: {workers: 2}\n")
    assert main(["validate", str(path)]) == 1
    assert "loader.workers" in caplog.text
