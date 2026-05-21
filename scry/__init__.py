"""Scry — Schneewolf Labs eval orchestrator.

Foundation only — see README. No working evaluation runner yet.
"""
from __future__ import annotations

from ._version import __version__
from .config import EvalConfig, BenchmarkSpec

__all__ = [
    "__version__",
    "EvalConfig",
    "BenchmarkSpec",
]
