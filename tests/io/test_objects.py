import numpy as np
import pytest
import shapely

from pathoinfer.io.objects import ObjectTable, RunInfo
from pathoinfer.stitch.base import StitchResult
from tests.stitch import test_detection as det
from tests.stitch import test_segmentation as seg


def test_run_info_records_slide_model_and_versions(seg_run: RunInfo) -> None:
    meta = seg_run.to_dict()
    assert meta["slide_id"] == "synthetic" and meta["task"] == "segmentation"
    assert meta["model_name"] == seg.card().name and meta["model_source"] == "models/tiny_seg"
    assert meta["tissue_area_um2"] == pytest.approx(seg.SIZE**2 * seg.SLIDE_MPP**2)
    assert meta["pathoinfer_version"] and meta["patchology_version"]
    # Background is never an object class.
    assert [c["id"] for c in meta["classes"]] == [1]


def test_colour_overrides_apply_and_unknown_classes_fail() -> None:
    run = RunInfo.create(det.card(), seg.grid(), "m", colors={"tumour": (1, 2, 3)})
    assert run.classes[0].color == (1, 2, 3) and run.classes[1].color == (0, 0, 255)
    with pytest.raises(ValueError, match="unknown classes"):
        RunInfo.create(det.card(), seg.grid(), "m", colors={"tumor": (1, 2, 3)})


def test_segmentation_scores_are_mean_probabilities(seg_objects: ObjectTable) -> None:
    assert len(seg_objects) == 2
    np.testing.assert_allclose(seg_objects.score, seg_objects.measurements["mean_probability"])
    assert set(seg_objects.measurements) == {"perimeter_um", "mean_probability"}
    np.testing.assert_allclose(
        seg_objects.area_um2, shapely.area(seg_objects.geometry) * seg.SLIDE_MPP**2, rtol=1e-6
    )


def test_detection_table_has_box_area_and_class_probabilities(det_run: RunInfo) -> None:
    boxes = det.squares(np.array([[1000.0, 1000.0], [2000.0, 2000.0]]), 40)
    found = StitchResult([], detections=det.stitch(boxes, det.card(probs=True)))
    for kind, geom_type in [("box", "Polygon"), ("point", "Point")]:
        table = ObjectTable.from_result(found, det_run, kind)  # type: ignore[arg-type]
        assert {g.geom_type for g in table.geometry} == {geom_type}
        np.testing.assert_allclose(table.area_um2, 40 * 40 * seg.SLIDE_MPP**2)
        first = int(np.flatnonzero(table.class_id == 0)[0])
        assert table.measurements["prob_tumour"][first] == pytest.approx(0.8)
        assert table.score[first] == pytest.approx(0.9)
    np.testing.assert_allclose(table.centroids(), found.detections.centroids())
