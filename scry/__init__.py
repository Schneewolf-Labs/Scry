"""Scry — Schneewolf Labs eval orchestrator.

v0.1: lm-eval-harness runner implemented for single-benchmark evaluation.
See README for roadmap to v1.0.
"""
from __future__ import annotations

from ._version import __version__
from .config import EvalConfig, BenchmarkSpec
from .runners import LmEvalHarnessRunner

__all__ = [
    "__version__",
    "EvalConfig",
    "BenchmarkSpec",
    "LmEvalHarnessRunner",
]
