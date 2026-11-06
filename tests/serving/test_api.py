"""The slide job API with fake engines, and once with the real one on the synthetic slide."""

import json
import time
from pathlib import Path
from typing import Any

import pytest

from pathoinfer.serving.api import JobManager, JobRequest, run_config, run_engine
from pathoinfer.tiling.synthetic import write_pyramid

from . import fakes

testclient = pytest.importorskip("fastapi.testclient")
CONFIG = {"task": "segmentation", "model": "tests/fixtures/tiny_dark"}


@pytest.fixture(scope="module")
def slide(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_pyramid(tmp_path_factory.mktemp("slides") / "slide_name-01.tiff")


def client(tmp_path: Path, runner: Any, max_jobs: int = 1) -> Any:
    from pathoinfer.serving.api import create_app

    return testclient.TestClient(create_app(tmp_path / "jobs", max_jobs, runner))


def wait(api: Any, job_id: str, statuses: tuple[str, ...], timeout: float = 60) -> dict[str, Any]:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        job: dict[str, Any] = api.get(f"/jobs/{job_id}").json()
        if job["status"] in statuses:
            return job
        time.sleep(0.1)
    raise AssertionError(f"job {job_id} stayed {job['status']}")


def submit(api: Any, slide: Path, config: dict[str, Any] | None = None) -> str:
    response = api.post("/jobs", json={"slide": str(slide), "config": config or CONFIG})
    assert response.status_code == 202, response.text
    job_id: str = response.json()["id"]
    return job_id


def test_a_job_runs_reports_progress_and_serves_its_outputs(tmp_path: Path, slide: Path) -> None:
    with client(tmp_path, fakes.quick) as api:
        job_id = submit(api, slide)
        job = wait(api, job_id, ("done", "failed"))
        assert job["status"] == "done", job["error"]
        assert (job["tiles_done"], job["tiles_total"]) == (4, 4)
        assert job["started_at"] and job["finished_at"]
        listed = api.get(f"/jobs/{job_id}/outputs").json()
        assert [o["name"] for o in listed] == ["nested/summary.parquet", "slide.geojson"]
        assert api.get(listed[1]["url"]).json()["type"] == "FeatureCollection"
        assert api.get(f"/jobs/{job_id}/outputs/../request.json").status_code == 404
        assert [j["id"] for j in api.get("/jobs").json()] == [job_id]
        assert api.get("/health").json() == {"status": "ok", "queued": 0, "running": 0}


def test_bad_requests_are_rejected_before_queueing(tmp_path: Path, slide: Path) -> None:
    with client(tmp_path, fakes.quick) as api:
        bad = api.post("/jobs", json={"slide": str(slide), "config": {"task": "tumour"}})
        assert bad.status_code == 422 and "task" in bad.text
        missing = api.post("/jobs", json={"slide": "/no/such.svs", "config": CONFIG})
        assert missing.status_code == 422 and "no slide" in missing.text
        assert api.get("/jobs").json() == []
        assert api.get("/jobs/abc123").status_code == 404


def test_a_failing_engine_fails_the_job_with_its_error(tmp_path: Path, slide: Path) -> None:
    with client(tmp_path, fakes.failing) as api:
        job = wait(api, submit(api, slide), ("done", "failed"))
        assert job["status"] == "failed" and "the model exploded" in job["error"]
        assert api.get(f"/jobs/{job['id']}/outputs").status_code == 409


def test_jobs_run_one_at_a_time_and_cancel_when_queued_or_running(
    tmp_path: Path, slide: Path
) -> None:
    with client(tmp_path, fakes.slow, max_jobs=1) as api:
        first, second = submit(api, slide), submit(api, slide)
        running = wait(api, first, ("running",))
        assert api.get(f"/jobs/{second}").json()["status"] == "queued"
        assert api.get("/health").json()["running"] == 1
        cancelled = api.post(f"/jobs/{second}/cancel").json()
        assert cancelled["status"] == "cancelled" and cancelled["started_at"] is None
        wait(api, first, ("running",))
        end = time.monotonic() + 30
        while api.get(f"/jobs/{first}").json()["tiles_done"] == 0 and time.monotonic() < end:
            time.sleep(0.1)
        assert api.get(f"/jobs/{first}").json()["tiles_done"] > 0
        stopped = api.post(f"/jobs/{first}/cancel").json()
        assert stopped["status"] == "cancelled" and running["started_at"]
        assert not (tmp_path / "jobs" / first / "result.json").exists()
        assert api.post(f"/jobs/{first}/cancel").status_code == 409
        time.sleep(0.5)
        assert api.get(f"/jobs/{second}").json()["status"] == "cancelled"  # never started
        assert not (tmp_path / "jobs" / second / "started").exists()


def test_two_slots_run_two_jobs_at_once(tmp_path: Path, slide: Path) -> None:
    with client(tmp_path, fakes.slow, max_jobs=2) as api:
        jobs = [submit(api, slide) for _ in range(3)]
        for job_id in jobs[:2]:
            wait(api, job_id, ("running",))
        assert api.get(f"/jobs/{jobs[2]}").json()["status"] == "queued"
        for job_id in jobs:
            api.post(f"/jobs/{job_id}/cancel")


def test_a_restart_fails_running_jobs_and_requeues_queued_ones(tmp_path: Path) -> None:
    root = tmp_path / "jobs"
    for job_id, status in [("aaa111", "running"), ("bbb222", "queued")]:
        (root / job_id).mkdir(parents=True)
        (root / job_id / "job.json").write_text(
            json.dumps({"id": job_id, "status": status, "slide": "s", "created_at": "2026"})
        )
    manager = JobManager(root, runner=fakes.quick)
    assert manager.get("aaa111").status == "failed"
    assert "restarted" in (manager.get("aaa111").error or "")
    assert manager.counts() == {"queued": 1, "running": 0}


def test_the_server_triton_url_reaches_jobs_unless_they_name_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PATHOINFER_TRITON_URL", "triton:8001")
    request = JobRequest(slide="s", config=CONFIG | {"outputs": {"gzip": False}})
    cfg = run_config(request, Path("s.svs"), Path("out"))
    assert cfg.runtime.triton_url == "triton:8001"
    assert cfg.outputs.dir == Path("out") and not cfg.outputs.gzip and not cfg.resume
    own = JobRequest(slide="s", config=CONFIG | {"runtime": {"triton_url": "other:8001"}})
    assert run_config(own, Path("s"), Path("o")).runtime.triton_url == "other:8001"


def test_the_real_engine_runs_a_job_on_the_synthetic_slide(tmp_path: Path, slide: Path) -> None:
    config = CONFIG | {"runtime": {"device": "cpu"}, "loader": {"num_workers": 0}}
    with client(tmp_path, run_engine) as api:
        job = wait(api, submit(api, slide, config), ("done", "failed"), timeout=300)
        assert job["status"] == "done", job["error"]
        names = [o["name"] for o in api.get(f"/jobs/{job['id']}/outputs").json()]
        assert "slide_name-01.geojson.gz" in names and "slide_name-01.objects.parquet" in names
        assert job["tiles_done"] == job["tiles_total"] > 0
