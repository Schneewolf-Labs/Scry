"""Eval-backend runner stubs.

Each backend (lm-eval-harness, VLMEvalKit, BFCL, custom batteries) will get a
concrete runner here. For the foundation release we only sketch the interface
the FastAPI layer will call into — so the API contract can be designed
without waiting for the harnesses to be integrated.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .config import BenchmarkSpec, EvalConfig


class BaseRunner(ABC):
    """Common interface for every eval backend.

    A runner is responsible for one *backend* (lm-eval-harness, VLMEvalKit,
    etc.). Within a single Scry job, the dispatcher selects the right runner
    per BenchmarkSpec.
    """

    backend_name: str = ""  # set on subclass

    @abstractmethod
    def can_handle(self, spec: BenchmarkSpec) -> bool:
        """Return True if this runner owns `spec.backend`."""

    @abstractmethod
    def run(
        self,
        spec: BenchmarkSpec,
        cfg: EvalConfig,
        progress_cb=None,
    ) -> dict[str, Any]:
        """Run one benchmark task. Return a results dict whose schema is the
        intersection across backends (at least: `score`, `task`, `n_samples`,
        plus arbitrary backend-specific extras under `details`).

        `progress_cb` is an optional `Callable[[int, int, str], None]` that
        receives (current_sample, total_samples, message); the FastAPI layer
        will hand in a callback that pushes WebSocket updates.
        """


# --- planned subclasses (NOT IMPLEMENTED) -----------------------------------

class LmEvalHarnessRunner(BaseRunner):
    """Wraps EleutherAI/lm-evaluation-harness via its Python API.

    Will install lm-eval-harness as an extra (`pip install scry[harness]`)
    rather than pinning a hard dependency, so users who only want VLM eval
    don't pay the cost.
    """
    backend_name = "lm-eval-harness"

    def can_handle(self, spec: BenchmarkSpec) -> bool:
        return spec.backend == "lm-eval-harness"

    def run(self, spec, cfg, progress_cb=None):
        raise NotImplementedError("Planned for Scry v0.1 — see ROADMAP")


class VlmEvalKitRunner(BaseRunner):
    """Wraps open-compass/VLMEvalKit. Used for the Artemis lineage."""
    backend_name = "vlm-eval-kit"

    def can_handle(self, spec: BenchmarkSpec) -> bool:
        return spec.backend == "vlm-eval-kit"

    def run(self, spec, cfg, progress_cb=None):
        raise NotImplementedError("Planned for Scry v0.4 — see ROADMAP")


class BfclRunner(BaseRunner):
    """Wraps Berkeley Function-Calling Leaderboard. Canonical tool-call eval."""
    backend_name = "bfcl"

    def can_handle(self, spec: BenchmarkSpec) -> bool:
        return spec.backend == "bfcl"

    def run(self, spec, cfg, progress_cb=None):
        raise NotImplementedError("Planned for Scry v0.4 — see ROADMAP")
