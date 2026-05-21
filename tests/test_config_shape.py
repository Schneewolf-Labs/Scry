"""Smoke tests for the EvalConfig / BenchmarkSpec Pydantic schemas.

These are the only things implemented in the foundation release — they exist
so contributors can iterate on the API contract independently of the runners.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from scry import BenchmarkSpec, EvalConfig


def test_minimal_eval_config():
    cfg = EvalConfig(
        base_model="schneewolflabs/A2",
        benchmarks=[BenchmarkSpec(backend="lm-eval-harness", task="ifeval")],
    )
    assert cfg.base_model == "schneewolflabs/A2"
    assert cfg.benchmarks[0].task == "ifeval"
    assert cfg.compare_to == []
    assert cfg.batch_size == 8


def test_paired_eval_config():
    cfg = EvalConfig(
        base_model="schneewolflabs/A2",
        compare_to=["schneewolflabs/A1"],
        benchmarks=[
            BenchmarkSpec(backend="bfcl", task="bfcl_v3_single", limit=100),
            BenchmarkSpec(backend="lm-eval-harness", task="mmlu-pro", num_fewshot=5),
        ],
    )
    assert cfg.compare_to == ["schneewolflabs/A1"]
    assert len(cfg.benchmarks) == 2
    assert cfg.benchmarks[0].limit == 100
    assert cfg.benchmarks[1].num_fewshot == 5


def test_unknown_backend_rejected():
    with pytest.raises(ValidationError):
        BenchmarkSpec(backend="totally-fake-backend", task="x")


def test_empty_benchmarks_rejected():
    with pytest.raises(ValidationError):
        EvalConfig(base_model="schneewolflabs/A2", benchmarks=[])
