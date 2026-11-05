from pathlib import Path

import polars as pl
import pytest
from pydantic import ValidationError

from pathoinfer.config import RunConfig, load_config

EXAMPLES = Path(__file__).parents[1] / "examples"
BASE = {"task": "detection", "model": "m.onnx", "slides": "s.svs"}


@pytest.mark.parametrize("task", ["segmentation", "detection", "classification"])
def test_example_config_loads(task: str) -> None:
    cfg = load_config(EXAMPLES / f"{task}.yaml")
    assert cfg.task == task
    assert cfg.tiling.icc


@pytest.mark.parametrize(
    "section",
    [
        {"tiling": {"slide_mpp": 0.25}},  # no override: slides without a resolution are refused
        {"tiling": {"tissue": {"methd": "otsu"}}},
        {"tiling": {"tissue": {"method": "nope"}}},
        {"tiling": {"overlap": 1.0}},
        {"loader": {"num_workers": -1}},
        {"outputs": {"format": "csv"}},
        {"class_colors": {"tumor": [256, 0, 0]}},
    ],
)
def test_bad_settings_are_rejected(section: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        RunConfig.model_validate(BASE | section)


def test_slides_from_paths_globs_and_list_files(tmp_path: Path) -> None:
    slides = [tmp_path / f"slide_name-0{i}.svs" for i in range(4)]
    for s in slides:
        s.touch()
    (tmp_path / "list.txt").write_text(f"{slides[0].name}\n\n{slides[1]}\n")
    pl.DataFrame({"path": [slides[2].name]}).write_csv(tmp_path / "list.csv")
    pl.DataFrame({"path": [str(slides[3])]}).write_parquet(tmp_path / "list.parquet")
    cfg = RunConfig.model_validate(
        BASE
        | {
            "slides": [
                str(tmp_path / "list.txt"),
                str(tmp_path / "list.csv"),
                str(tmp_path / "list.parquet"),
                str(tmp_path / "*.svs"),
            ]
        }
    )
    assert cfg.slide_paths() == slides
    assert RunConfig.model_validate(BASE | {"slides": str(slides[1])}).slide_paths() == [slides[1]]


def test_settings_hash_tracks_what_changes_outputs() -> None:
    base = RunConfig.model_validate(BASE)
    same = base.model_copy(
        update={
            "slides": ["other.svs"],
            "resume": False,
            "loader": base.loader.model_copy(update={"num_workers": 0}),
        }
    )
    assert base.settings_hash("{}") == same.settings_hash("{}")
    changed = RunConfig.model_validate(BASE | {"detection": {"nms_iou": 0.3}})
    assert changed.settings_hash("{}") != base.settings_hash("{}")
    assert base.settings_hash('{"version": "2"}') != base.settings_hash("{}")
