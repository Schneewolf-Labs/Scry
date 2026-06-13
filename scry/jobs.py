"""Job queue, persistence, and progress fan-out for Scry (v0.2).

Single-GPU model: one daemon worker thread drains a FIFO queue, so at most
one eval job touches the GPU at a time. Job metadata and results persist in
sqlite (``./data/eval_jobs.db`` by default) so history survives restarts.
Live progress events fan out to in-process subscribers — that is what the
WebSocket layer in `scry.app` consumes.

Mirrors Merlina's job_manager/job_queue split, collapsed into one module
since Scry's needs are smaller.
"""
from __future__ import annotations

import json
import logging
import queue
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from .config import EvalConfig
from .runners import BaseRunner, default_runners, get_runner

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STATUSES = frozenset(
    {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}
)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Job:
    """One eval request and everything we know about it."""

    id: str
    status: JobStatus
    config: dict[str, Any]
    results: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    progress: Optional[dict[str, Any]] = None
    created_at: str = ""
    started_at: Optional[str] = None
    finished_at: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """API-facing shape. Secrets in the config are masked."""
        config = dict(self.config)
        if config.get("hf_token"):
            config["hf_token"] = "***"
        return {
            "id": self.id,
            "status": self.status.value,
            "config": config,
            "results": self.results,
            "error": self.error,
            "progress": self.progress,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    config TEXT NOT NULL,
    results TEXT,
    error TEXT,
    progress TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT
)
"""

_JSON_FIELDS = {"config", "results", "progress"}
_UPDATABLE_FIELDS = {
    "status", "results", "error", "progress", "started_at", "finished_at"
}


class JobStore:
    """sqlite-backed job persistence.

    Thread-safe: a single connection shared across threads, serialized by a
    lock. Writes are tiny (one row per state change), so contention is a
    non-issue at Scry's scale.
    """

    def __init__(self, db_path: str | Path = "data/eval_jobs.db") -> None:
        self.db_path = str(db_path)
        parent = Path(self.db_path).parent
        if parent and not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock, self._conn:
            self._conn.execute(_SCHEMA)

    def create(self, config: EvalConfig) -> Job:
        job = Job(
            id=uuid.uuid4().hex[:12],
            status=JobStatus.QUEUED,
            config=config.model_dump(),
            created_at=_utcnow(),
        )
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO jobs (id, status, config, created_at) "
                "VALUES (?, ?, ?, ?)",
                (job.id, job.status.value, json.dumps(job.config), job.created_at),
            )
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
        return self._from_row(row) if row else None

    def list(self, limit: int = 50, status: Optional[JobStatus] = None) -> list[Job]:
        """Most recent jobs first."""
        sql = "SELECT * FROM jobs"
        params: list[Any] = []
        if status is not None:
            sql += " WHERE status = ?"
            params.append(status.value)
        sql += " ORDER BY created_at DESC, rowid DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [self._from_row(row) for row in rows]

    def update(self, job_id: str, **fields: Any) -> None:
        unknown = set(fields) - _UPDATABLE_FIELDS
        if unknown:
            raise ValueError(f"Cannot update job fields: {sorted(unknown)}")
        columns, values = [], []
        for name, value in fields.items():
            if name in _JSON_FIELDS and value is not None:
                value = json.dumps(value)
            elif isinstance(value, JobStatus):
                value = value.value
            columns.append(f"{name} = ?")
            values.append(value)
        values.append(job_id)
        with self._lock, self._conn:
            self._conn.execute(
                f"UPDATE jobs SET {', '.join(columns)} WHERE id = ?", values
            )

    @staticmethod
    def _from_row(row: sqlite3.Row) -> Job:
        def load(name: str) -> Any:
            raw = row[name]
            return json.loads(raw) if raw is not None else None

        return Job(
            id=row["id"],
            status=JobStatus(row["status"]),
            config=load("config"),
            results=load("results"),
            error=row["error"],
            progress=load("progress"),
            created_at=row["created_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
        )

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class JobManager:
    """FIFO queue + single worker thread + event fan-out.

    submit() persists the job and enqueues it; the worker runs each
    benchmark in the job's config through the matching runner. stop() on a
    queued job cancels it outright; on a running job it takes effect after
    the current benchmark (runners can't be interrupted mid-eval).
    """

    def __init__(
        self,
        db_path: str | Path = "data/eval_jobs.db",
        runners: Optional[list[BaseRunner]] = None,
    ) -> None:
        self.store = JobStore(db_path)
        self.runners = runners if runners is not None else default_runners()
        self._queue: queue.Queue[str] = queue.Queue()
        self._stop_flags: dict[str, threading.Event] = {}
        self._subscribers: dict[str, list[queue.Queue]] = {}
        self._lock = threading.Lock()
        self._shutdown = threading.Event()
        self._recover()
        self._worker = threading.Thread(
            target=self._work_loop, name="scry-job-worker", daemon=True
        )
        self._worker.start()

    # ------------------------------------------------------------------ API

    def submit(self, config: EvalConfig) -> Job:
        job = self.store.create(config)
        with self._lock:
            self._stop_flags[job.id] = threading.Event()
        self._queue.put(job.id)
        logger.info(f"Job {job.id} queued: {config.base_model}, "
                    f"{len(config.benchmarks)} benchmark(s)")
        return job

    def stop(self, job_id: str) -> Job:
        """Cancel a queued job, or request a running one to halt after the
        current benchmark. Raises KeyError for unknown ids; stopping an
        already-finished job is a no-op."""
        job = self.store.get(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.status in TERMINAL_STATUSES:
            return job
        # Set the flag first so a job popped between our status check and the
        # update still sees the stop request.
        with self._lock:
            flag = self._stop_flags.get(job_id)
        if flag is not None:
            flag.set()
        if job.status is JobStatus.QUEUED:
            self._finish(job_id, JobStatus.CANCELLED)
        job = self.store.get(job_id)
        assert job is not None
        return job

    def subscribe(self, job_id: str) -> queue.Queue:
        """Get a queue that receives this job's progress events as dicts."""
        q: queue.Queue = queue.Queue()
        with self._lock:
            self._subscribers.setdefault(job_id, []).append(q)
        return q

    def unsubscribe(self, job_id: str, q: queue.Queue) -> None:
        with self._lock:
            subscribers = self._subscribers.get(job_id, [])
            if q in subscribers:
                subscribers.remove(q)
            if not subscribers:
                self._subscribers.pop(job_id, None)

    def shutdown(self, timeout: float = 5.0) -> None:
        self._shutdown.set()
        self._worker.join(timeout=timeout)
        self.store.close()

    # --------------------------------------------------------------- worker

    def _recover(self) -> None:
        """Reconcile sqlite state left over from a previous process: jobs
        caught mid-run are failed (their GPU work is gone); queued jobs are
        re-enqueued in their original order."""
        for job in reversed(self.store.list(limit=1_000_000, status=JobStatus.RUNNING)):
            logger.warning(f"Job {job.id} was running at shutdown; marking failed")
            self._finish(
                job.id, JobStatus.FAILED,
                error="Interrupted by server restart",
            )
        for job in reversed(self.store.list(limit=1_000_000, status=JobStatus.QUEUED)):
            logger.info(f"Re-enqueuing job {job.id} from previous session")
            with self._lock:
                self._stop_flags[job.id] = threading.Event()
            self._queue.put(job.id)

    def _work_loop(self) -> None:
        while not self._shutdown.is_set():
            try:
                job_id = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._run_job(job_id)
            except Exception:
                logger.exception(f"Unexpected error processing job {job_id}")
            finally:
                self._queue.task_done()
                with self._lock:
                    self._stop_flags.pop(job_id, None)

    def _run_job(self, job_id: str) -> None:
        job = self.store.get(job_id)
        if job is None or job.status is not JobStatus.QUEUED:
            return  # cancelled (or vanished) while waiting in the queue
        with self._lock:
            stop_flag = self._stop_flags.get(job_id, threading.Event())

        cfg = EvalConfig.model_validate(job.config)
        self.store.update(job_id, status=JobStatus.RUNNING, started_at=_utcnow())
        self._publish(job_id, {"type": "status", "job_id": job_id,
                               "status": JobStatus.RUNNING.value})

        total = len(cfg.benchmarks)
        benchmark_results: list[dict[str, Any]] = []
        error: Optional[str] = None
        cancelled = False

        for index, spec in enumerate(cfg.benchmarks):
            if stop_flag.is_set():
                cancelled = True
                break
            self.store.update(job_id, progress={
                "task": spec.task, "backend": spec.backend,
                "index": index, "total_benchmarks": total,
            })
            self._publish(job_id, {
                "type": "benchmark_started", "job_id": job_id,
                "task": spec.task, "backend": spec.backend,
                "index": index, "total": total,
            })

            # Per-sample progress is fanned out live but not persisted —
            # sqlite sees one write per benchmark, not one per sample.
            def progress_cb(current: int, total_samples: int, message: str,
                            _task: str = spec.task) -> None:
                self._publish(job_id, {
                    "type": "progress", "job_id": job_id, "task": _task,
                    "current": current, "total": total_samples,
                    "message": message,
                })

            try:
                runner = get_runner(spec, self.runners)
                result = runner.run(spec, cfg, progress_cb)
            except Exception as e:
                error = f"{spec.task}: {e}"
                logger.exception(f"Job {job_id} benchmark '{spec.task}' failed")
                break

            benchmark_results.append(result)
            self._publish(job_id, {
                "type": "benchmark_completed", "job_id": job_id,
                "task": spec.task, "score": result.get("score"),
                "index": index, "total": total,
            })

        results = {
            "job_id": job_id,
            "base_model": cfg.base_model,
            "benchmarks": benchmark_results,
            "summary": {r["task"]: r.get("score") for r in benchmark_results},
        }
        if error is not None:
            status = JobStatus.FAILED
        elif cancelled:
            status = JobStatus.CANCELLED
        else:
            status = JobStatus.COMPLETED
        self._finish(job_id, status, results=results, error=error)
        logger.info(f"Job {job_id} {status.value}")

    def _finish(self, job_id: str, status: JobStatus, *,
                results: Optional[dict[str, Any]] = None,
                error: Optional[str] = None) -> None:
        fields: dict[str, Any] = {
            "status": status, "finished_at": _utcnow(), "error": error,
        }
        if results is not None:
            fields["results"] = results
        self.store.update(job_id, **fields)
        event: dict[str, Any] = {
            "type": "status", "job_id": job_id, "status": status.value,
        }
        if results is not None:
            event["summary"] = results["summary"]
        if error is not None:
            event["error"] = error
        self._publish(job_id, event)

    def _publish(self, job_id: str, event: dict[str, Any]) -> None:
        with self._lock:
            subscribers = list(self._subscribers.get(job_id, []))
        for q in subscribers:
            q.put(event)
