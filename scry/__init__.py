"""Scry — Schneewolf Labs eval orchestrator.

v0.2: job queue + FastAPI server + WebSocket progress; multi-benchmark per
request. See README for roadmap to v1.0.
"""
from __future__ import annotations

from ._version import __version__
from .config import EvalConfig, BenchmarkSpec
from .jobs import Job, JobManager, JobStatus, JobStore
from .runners import LmEvalHarnessRunner, default_runners, get_runner

__all__ = [
    "__version__",
    "EvalConfig",
    "BenchmarkSpec",
    "Job",
    "JobManager",
    "JobStatus",
    "JobStore",
    "LmEvalHarnessRunner",
    "default_runners",
    "get_runner",
]
