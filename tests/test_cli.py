"""Tests for the scry-eval CLI.

The runner is stubbed out so these exercise argument parsing, config
construction, and output handling without lm-eval-harness installed.
"""
import json
import sys

import pytest

from scry import cli


class DummyRunner:
    """Stands in for LmEvalHarnessRunner; echoes back what it was given."""

    def run(self, spec, cfg, progress_cb=None):
        return {
            "task": spec.task,
            "model": cfg.base_model,
            "num_fewshot": spec.num_fewshot,
            "limit": spec.limit,
            "score": 0.5,
            "metrics": {"acc,none": 0.5},
        }


def _invoke(monkeypatch, argv, runner_cls=DummyRunner):
    monkeypatch.setattr(cli, "LmEvalHarnessRunner", runner_cls)
    monkeypatch.setattr(sys, "argv", ["scry-eval"] + argv)
    cli.main()


def test_cli_minimal_invocation(monkeypatch, capsys):
    """The documented minimal invocation must build a valid EvalConfig."""
    _invoke(monkeypatch, ["--model", "EleutherAI/pythia-70m", "--task", "arc_easy"])

    results = json.loads(capsys.readouterr().out)
    assert results["model"] == "EleutherAI/pythia-70m"
    assert results["task"] == "arc_easy"
    assert results["num_fewshot"] is None  # unset -> task default
    assert results["score"] == 0.5


def test_cli_all_flags(monkeypatch, capsys):
    _invoke(
        monkeypatch,
        [
            "--model", "EleutherAI/pythia-70m",
            "--task", "arc_easy",
            "--num-fewshot", "5",
            "--limit", "20",
            "--batch-size", "4",
            "--max-length", "2048",
        ],
    )

    results = json.loads(capsys.readouterr().out)
    assert results["num_fewshot"] == 5
    assert results["limit"] == 20


def test_cli_writes_output_file(monkeypatch, tmp_path):
    out_file = tmp_path / "results.json"
    _invoke(
        monkeypatch,
        ["--model", "m", "--task", "t", "--output", str(out_file)],
    )

    results = json.loads(out_file.read_text())
    assert results["score"] == 0.5


def test_cli_exits_nonzero_without_score(monkeypatch):
    class NoScoreRunner:
        def run(self, spec, cfg, progress_cb=None):
            return {"score": None}

    with pytest.raises(SystemExit) as exc:
        _invoke(monkeypatch, ["--model", "m", "--task", "t"], runner_cls=NoScoreRunner)
    assert exc.value.code == 1
