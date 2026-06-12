"""FastAPI surface for Scry (v0.2).

POST /eval enqueues a job (202 + job_id) carrying one or more benchmarks;
a single worker thread drains the queue so one eval owns the GPU at a time.
Progress streams over WS /ws/jobs/{job_id}; results and job history are
served from the sqlite store.

Mirrors Merlina's app.py shape:
- POST /eval -> 202 + job_id
- GET /jobs / GET /jobs/{id} / POST /jobs/{id}/stop
- GET /results/{id}
- WS /ws/jobs/{id} for live progress

Requires the `api` extra: pip install "scry[api]". Run with `scry-server`.
"""
from __future__ import annotations

import asyncio
import queue
from contextlib import asynccontextmanager
from typing import Any, Optional

try:
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "FastAPI is required for the Scry API server. "
        "Install with: pip install 'scry[api]'"
    ) from e

from ._version import __version__
from .config import EvalConfig
from .jobs import TERMINAL_STATUSES, JobManager

_TERMINAL_VALUES = {status.value for status in TERMINAL_STATUSES}


def _next_event(q: queue.Queue) -> Optional[dict[str, Any]]:
    """Blocking get with a short timeout so the WebSocket loop can notice
    terminal jobs / disconnects instead of parking a thread forever."""
    try:
        return q.get(timeout=0.5)
    except queue.Empty:
        return None


def create_app(
    manager: Optional[JobManager] = None,
    db_path: str = "data/eval_jobs.db",
) -> FastAPI:
    """Build the Scry API app.

    Pass an existing JobManager (tests do) or let the app own one backed by
    `db_path`. An owned manager is shut down with the app.
    """
    owns_manager = manager is None
    mgr = manager if manager is not None else JobManager(db_path=db_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        if owns_manager:
            mgr.shutdown()

    app = FastAPI(
        title="Scry",
        description="Eval orchestrator for HuggingFace models (Schneewolf Labs)",
        version=__version__,
        lifespan=lifespan,
    )
    app.state.manager = mgr

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.post("/eval", status_code=202)
    async def submit_eval(config: EvalConfig) -> dict[str, str]:
        job = mgr.submit(config)
        return {"job_id": job.id, "status": job.status.value}

    @app.get("/jobs")
    async def list_jobs(limit: int = 50) -> dict[str, Any]:
        return {"jobs": [job.to_dict() for job in mgr.store.list(limit=limit)]}

    @app.get("/jobs/{job_id}")
    async def get_job(job_id: str) -> dict[str, Any]:
        job = mgr.store.get(job_id)
        if job is None:
            raise HTTPException(404, f"Job '{job_id}' not found")
        return job.to_dict()

    @app.post("/jobs/{job_id}/stop")
    async def stop_job(job_id: str) -> dict[str, str]:
        try:
            job = mgr.stop(job_id)
        except KeyError:
            raise HTTPException(404, f"Job '{job_id}' not found")
        if job.status.value in _TERMINAL_VALUES:
            detail = "Job stopped."
        else:
            detail = ("Stop requested; the job will halt after the current "
                      "benchmark (runners cannot be interrupted mid-eval).")
        return {"job_id": job.id, "status": job.status.value, "detail": detail}

    @app.get("/results/{job_id}")
    async def get_results(job_id: str) -> dict[str, Any]:
        job = mgr.store.get(job_id)
        if job is None:
            raise HTTPException(404, f"Job '{job_id}' not found")
        if job.status.value not in _TERMINAL_VALUES:
            raise HTTPException(
                409, f"Job '{job_id}' is still {job.status.value}; "
                     "results are available once it finishes"
            )
        return {
            "job_id": job.id,
            "status": job.status.value,
            "results": job.results,
            "error": job.error,
        }

    @app.websocket("/ws/jobs/{job_id}")
    async def job_progress(websocket: WebSocket, job_id: str) -> None:
        job = mgr.store.get(job_id)
        if job is None:
            await websocket.close(code=4404, reason=f"Job '{job_id}' not found")
            return
        await websocket.accept()
        events = mgr.subscribe(job_id)
        loop = asyncio.get_running_loop()
        try:
            await websocket.send_json({"type": "snapshot", "job": job.to_dict()})
            if job.status.value in _TERMINAL_VALUES:
                await websocket.close()
                return
            while True:
                event = await loop.run_in_executor(None, _next_event, events)
                if event is None:
                    # The terminal event may have fired between the snapshot
                    # read and our subscribe; re-check instead of waiting on
                    # an event that will never come.
                    current = mgr.store.get(job_id)
                    if current is None or current.status.value in _TERMINAL_VALUES:
                        await websocket.send_json(
                            {"type": "snapshot", "job": current.to_dict()}
                            if current else
                            {"type": "status", "job_id": job_id, "status": "failed"}
                        )
                        break
                    continue
                await websocket.send_json(event)
                if (event.get("type") == "status"
                        and event.get("status") in _TERMINAL_VALUES):
                    break
            await websocket.close()
        except WebSocketDisconnect:
            pass
        finally:
            mgr.unsubscribe(job_id, events)

    return app
