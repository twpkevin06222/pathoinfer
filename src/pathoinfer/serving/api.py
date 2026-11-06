"""Slide job API (FastAPI, ``[serve]`` extra): jobs, not single requests.

``POST /jobs`` takes a slide (a path on the server, ``file://`` or ``http(s)://`` URI)
and a run config, and returns a job id at once. Jobs run the engine (``run_slide``) in
their own worker processes, at most ``max_jobs`` at a time; the rest wait in order.

| Endpoint | Does |
|---|---|
| ``POST /jobs`` | Submit; ``202`` with the job. The config is validated now, so a bad one is a ``422`` |
| ``GET /jobs`` | Every job, newest first |
| ``GET /jobs/{id}`` | Status (``queued``, ``running``, ``done``, ``failed``, ``cancelled``) and progress in tiles |
| ``GET /jobs/{id}/outputs`` | The output files of a finished job |
| ``GET /jobs/{id}/outputs/{name}`` | One output file |
| ``POST /jobs/{id}/cancel`` | Dequeue a waiting job or stop a running one |
| ``GET /health`` | Liveness and the queue |

State lives on disk under ``jobs_dir``: ``<id>/job.json`` (written only by the API
process), ``<id>/progress.json`` and ``<id>/result.json`` (written only by the job's
worker), ``<id>/outputs/``. A job left running by a previous server process is marked
failed when the server starts; queued jobs are queued again.

The run config is a ``RunConfig`` without ``slides``, ``outputs.dir`` and ``resume``,
which the server sets. With ``PATHOINFER_TRITON_URL`` set, jobs whose config names no
``runtime.triton_url`` run their model on that Triton server.
"""

import json
import multiprocessing
import os
import threading
import time
import traceback
import uuid
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from loguru import logger
from pydantic import BaseModel, ConfigDict, ValidationError

from pathoinfer._extras import extra, require
from pathoinfer.config import RunConfig

Status = Literal["queued", "running", "done", "failed", "cancelled"]
Progress = Callable[[int, int], None]
# (request, job folder, progress) -> output files; runs in the job's worker process.
Runner = Callable[["JobRequest", Path, Progress], list[Path]]
FINISHED = ("done", "failed", "cancelled")
POLL_S = 0.2
STOP_GRACE_S = 10.0


class JobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slide: str
    config: dict[str, Any]


class Job(BaseModel):
    id: str
    status: Status
    slide: str
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    tiles_done: int = 0
    tiles_total: int | None = None
    error: str | None = None
    outputs: list[str] = []


def run_config(request: JobRequest, slide: Path, out: Path) -> RunConfig:
    """The ``RunConfig`` a job runs: the request's, with the server's slide, output
    folder, no resume, and the server's Triton URL unless the request names one."""
    data = dict(request.config)
    data["slides"] = str(slide)
    data["outputs"] = {**data.get("outputs", {}), "dir": str(out)}
    data["resume"] = False
    triton = os.environ.get("PATHOINFER_TRITON_URL")
    runtime = dict(data.get("runtime", {}))
    if triton and not runtime.get("triton_url"):
        data["runtime"] = runtime | {"triton_url": triton}
    return RunConfig.model_validate(data)


def run_engine(request: JobRequest, folder: Path, progress: Progress) -> list[Path]:
    """The default runner: fetch the slide, load the model, run ``run_slide``."""
    from pathoinfer.engine.pipeline import load, run_slide

    slide = fetch_slide(request.slide, folder / "slide")
    cfg = run_config(request, slide, folder / "outputs")
    result = run_slide(cfg, slide, load(cfg), on_progress=progress)
    return list(result.outputs)


def fetch_slide(uri: str, into: Path) -> Path:
    """A local path or ``file://`` URI as is; ``http(s)://`` downloaded into ``into``."""
    if uri.startswith(("http://", "https://")):
        httpx = require("httpx", extra("serve"))
        into.mkdir(parents=True, exist_ok=True)
        target = into / (uri.rstrip("/").rsplit("/", 1)[-1].split("?")[0] or "slide")
        with httpx.stream("GET", uri, follow_redirects=True, timeout=None) as response:
            response.raise_for_status()
            with open(target, "wb") as f:
                f.writelines(response.iter_bytes(1 << 20))
        return target
    path = Path(uri.removeprefix("file://"))
    if not path.is_file():
        raise FileNotFoundError(f"no slide at {uri}")
    return path


class JobManager:
    """Queue, worker processes and on-disk state of the jobs under ``root``."""

    def __init__(self, root: Path, max_jobs: int = 1, runner: Runner = run_engine) -> None:
        self.root, self.max_jobs, self.runner = root, max_jobs, runner
        root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._queue: deque[str] = deque()
        self._running: dict[str, multiprocessing.process.BaseProcess] = {}
        self._stop = threading.Event()
        self._recover()
        self._thread = threading.Thread(target=self._loop, name="jobs", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def shutdown(self) -> None:
        """Stop the scheduler and the running jobs (marked failed: the server stopped)."""
        self._stop.set()
        self._thread.join(timeout=5)
        with self._lock:
            for job_id, process in list(self._running.items()):
                _stop(process)
                self._finish(job_id, "failed", error="the server stopped")
            self._running.clear()

    def submit(self, request: JobRequest) -> Job:
        """Validate and queue. Raises ValueError for a bad config or a missing slide."""
        try:
            run_config(request, Path("slide"), Path("outputs"))
        except ValidationError as e:
            raise ValueError(str(e)) from e
        if not request.slide.startswith(("http://", "https://")):
            fetch_slide(request.slide, self.root)  # raises if the file is missing
        job = Job(id=uuid.uuid4().hex[:12], status="queued", slide=request.slide, created_at=_now())
        folder = self.root / job.id
        folder.mkdir()
        (folder / "request.json").write_text(request.model_dump_json())
        self._write(job)
        with self._lock:
            self._queue.append(job.id)
        logger.info("job {} queued: {}", job.id, request.slide)
        return job

    def get(self, job_id: str) -> Job:
        path = self.root / job_id / "job.json"
        if not job_id.isalnum() or not path.is_file():
            raise KeyError(job_id)
        job = Job.model_validate_json(path.read_text())
        progress = self.root / job_id / "progress.json"
        if job.status == "running" and progress.is_file():
            done, total = json.loads(progress.read_text())
            job = job.model_copy(update={"tiles_done": done, "tiles_total": total})
        return job

    def list(self) -> list[Job]:
        jobs = [self.get(p.parent.name) for p in self.root.glob("*/job.json")]
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)

    def cancel(self, job_id: str) -> Job:
        """Raises KeyError for an unknown job, ValueError for a finished one."""
        job = self.get(job_id)
        if job.status in FINISHED:
            raise ValueError(f"job {job_id} is already {job.status}")
        with self._lock:
            if job_id in self._queue:
                self._queue.remove(job_id)
            process = self._running.pop(job_id, None)
            if process is not None:
                _stop(process)
            self._finish(job_id, "cancelled")
        logger.info("job {} cancelled", job_id)
        return self.get(job_id)

    def counts(self) -> dict[str, int]:
        with self._lock:
            return {"queued": len(self._queue), "running": len(self._running)}

    def _loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                for job_id, process in list(self._running.items()):
                    if not process.is_alive():
                        del self._running[job_id]
                        self._reap(job_id, process)
                while self._queue and len(self._running) < self.max_jobs:
                    self._launch(self._queue.popleft())
            time.sleep(POLL_S)

    def _launch(self, job_id: str) -> None:
        folder = self.root / job_id
        process = multiprocessing.get_context("spawn").Process(
            target=_work, args=(self.runner, folder), name=f"job-{job_id}", daemon=False
        )
        process.start()
        self._running[job_id] = process
        self._write(self.get(job_id).model_copy(update={"status": "running", "started_at": _now()}))
        logger.info("job {} started (pid {})", job_id, process.pid)

    def _reap(self, job_id: str, process: multiprocessing.process.BaseProcess) -> None:
        folder = self.root / job_id
        result = folder / "result.json"
        if process.exitcode == 0 and result.is_file():
            outputs = json.loads(result.read_text())["outputs"]
            self._finish(job_id, "done", outputs=outputs)
        else:
            error_file = folder / "error.txt"
            error = (
                error_file.read_text() if error_file.is_file() else f"exit code {process.exitcode}"
            )
            self._finish(job_id, "failed", error=error)
        logger.info("job {} {}", job_id, self.get(job_id).status)

    def _finish(self, job_id: str, status: Status, **fields: Any) -> None:
        job = self.get(job_id)
        progress = {"tiles_done": job.tiles_done, "tiles_total": job.tiles_total}
        self._write(
            job.model_copy(update={"status": status, "finished_at": _now(), **progress, **fields})
        )

    def _recover(self) -> None:
        for path in sorted(self.root.glob("*/job.json")):
            job = Job.model_validate_json(path.read_text())
            if job.status == "running":
                self._write(
                    job.model_copy(update={"status": "failed", "error": "the server restarted"})
                )
            elif job.status == "queued":
                self._queue.append(job.id)

    def _write(self, job: Job) -> None:
        path = self.root / job.id / "job.json"
        partial = path.with_suffix(".partial")
        partial.write_text(job.model_dump_json(indent=2))
        os.replace(partial, path)


def _work(runner: Runner, folder: Path) -> None:
    """A job's worker process: run, then write result.json or error.txt."""
    request = JobRequest.model_validate_json((folder / "request.json").read_text())
    last = 0.0

    def progress(done: int, total: int) -> None:
        nonlocal last
        now = time.monotonic()
        if now - last >= 0.5 or done >= total:
            last = now
            partial = folder / "progress.partial"
            partial.write_text(json.dumps([done, total]))
            os.replace(partial, folder / "progress.json")

    try:
        outputs = runner(request, folder, progress)
        out = folder / "outputs"
        names = sorted(str(Path(p).resolve().relative_to(out.resolve())) for p in outputs)
        (folder / "result.json").write_text(json.dumps({"outputs": names}))
    except BaseException as e:
        (folder / "error.txt").write_text(f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}")
        raise SystemExit(1) from e


def _stop(process: multiprocessing.process.BaseProcess) -> None:
    process.terminate()
    process.join(STOP_GRACE_S)
    if process.is_alive():
        process.kill()
        process.join()


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def create_app(jobs_dir: Path, max_jobs: int = 1, runner: Runner = run_engine) -> Any:
    """The FastAPI app; the job manager starts with the app and stops with it."""
    require("fastapi", extra("serve"))
    from contextlib import asynccontextmanager

    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse

    manager = JobManager(jobs_dir, max_jobs, runner)

    @asynccontextmanager
    async def lifespan(app: Any) -> Any:
        manager.start()
        yield
        manager.shutdown()

    app = FastAPI(title="pathoinfer", lifespan=lifespan)
    app.state.jobs = manager

    def known(job_id: str) -> Job:
        try:
            return manager.get(job_id)
        except KeyError:
            raise HTTPException(404, f"no job {job_id}") from None

    @app.post("/jobs", status_code=202)
    def submit(request: JobRequest) -> Job:
        try:
            return manager.submit(request)
        except (ValueError, FileNotFoundError) as e:
            raise HTTPException(422, str(e)) from e

    @app.get("/jobs")
    def jobs() -> list[Job]:
        return manager.list()

    @app.get("/jobs/{job_id}")
    def job(job_id: str) -> Job:
        return known(job_id)

    @app.post("/jobs/{job_id}/cancel")
    def cancel(job_id: str) -> Job:
        known(job_id)
        try:
            return manager.cancel(job_id)
        except ValueError as e:
            raise HTTPException(409, str(e)) from e

    @app.get("/jobs/{job_id}/outputs")
    def outputs(job_id: str) -> list[dict[str, Any]]:
        found = known(job_id)
        if found.status != "done":
            raise HTTPException(409, f"job {job_id} is {found.status}")
        folder = jobs_dir / job_id / "outputs"
        return [
            {"name": n, "bytes": (folder / n).stat().st_size, "url": f"/jobs/{job_id}/outputs/{n}"}
            for n in found.outputs
        ]

    @app.get("/jobs/{job_id}/outputs/{name:path}")
    def output(job_id: str, name: str) -> Any:
        found = known(job_id)
        if name not in found.outputs:
            raise HTTPException(404, f"job {job_id} has no output {name}")
        return FileResponse(jobs_dir / job_id / "outputs" / name)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", **manager.counts()}

    return app
