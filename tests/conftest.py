"""Shared test helpers: a controllable fake runner and wait utilities."""
from __future__ import annotations

import time
from typing import Callable, Optional

from scry.config import BenchmarkSpec, EvalConfig
from scry.runners import BaseRunner


class FakeRunner(BaseRunner):
    """In-memory runner for backend='custom'.

    `gate` (a threading.Event) blocks run() until set, letting tests hold a
    job mid-flight; `started` is set when run() begins so tests can wait for
    that moment. Tasks listed in `fail_tasks` raise RuntimeError.
    """

    backend_name = "custom"

    def __init__(self, score: float = 0.5, fail_tasks: tuple[str, ...] = (),
                 gate=None, started=None) -> None:
        self.score = score
        self.fail_tasks = set(fail_tasks)
        self.gate = gate
        self.started = started
        self.calls: list[str] = []

    def can_handle(self, spec: BenchmarkSpec) -> bool:
        return spec.backend == "custom"

    def run(self, spec, cfg, progress_cb=None):
        self.calls.append(spec.task)
        if self.started is not None:
            self.started.set()
        if self.gate is not None:
            assert self.gate.wait(timeout=10), "test gate never released"
        if spec.task in self.fail_tasks:
            raise RuntimeError(f"boom: {spec.task}")
        if progress_cb is not None:
            progress_cb(1, 2, f"{spec.task}: halfway")
            progress_cb(2, 2, f"{spec.task}: done")
        return {
            "task": spec.task,
            "backend": self.backend_name,
            "model": cfg.base_model,
            "score": self.score,
            "n_samples": 2,
        }


def make_config(*tasks: str, **cfg_kwargs) -> EvalConfig:
    specs = [
        BenchmarkSpec(backend="custom", task=task)
        for task in (tasks or ("task_a",))
    ]
    return EvalConfig(
        base_model="schneewolflabs/test-model", benchmarks=specs, **cfg_kwargs
    )


def wait_for(predicate: Callable, timeout: float = 10.0,
             interval: float = 0.02):
    """Poll until `predicate()` is truthy and return its value."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    raise AssertionError("condition not met within timeout")
