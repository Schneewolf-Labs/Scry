"""Tests for Scry runners.

v0.1: Tests for lm-eval-harness runner.
"""
import sys
import types

import pytest
from scry.config import EvalConfig, BenchmarkSpec
from scry.runners import LmEvalHarnessRunner


class TestLmEvalHarnessRunner:
    """Tests for the lm-eval-harness runner."""

    def test_can_handle_correct_backend(self):
        """Runner should handle lm-eval-harness backend."""
        runner = LmEvalHarnessRunner()
        spec = BenchmarkSpec(backend="lm-eval-harness", task="arc_easy")
        assert runner.can_handle(spec) is True

    def test_cannot_handle_other_backends(self):
        """Runner should not handle other backends."""
        runner = LmEvalHarnessRunner()

        spec_vlm = BenchmarkSpec(backend="vlm-eval-kit", task="mme")
        spec_bfcl = BenchmarkSpec(backend="bfcl", task="single_function_call")
        spec_custom = BenchmarkSpec(backend="custom", task="my_task")

        assert runner.can_handle(spec_vlm) is False
        assert runner.can_handle(spec_bfcl) is False
        assert runner.can_handle(spec_custom) is False

    def test_extract_primary_score_acc(self):
        """Should extract 'acc' as primary score when available."""
        runner = LmEvalHarnessRunner()
        metrics = {"acc": 0.85, "acc_stderr": 0.02, "f1": 0.82}
        assert runner._extract_primary_score(metrics) == 0.85

    def test_extract_primary_score_fallback(self):
        """Should fallback to first numeric metric."""
        runner = LmEvalHarnessRunner()
        metrics = {"perplexity": 12.5, "perplexity_stderr": 0.3}
        assert runner._extract_primary_score(metrics) == 12.5

    def test_extract_primary_score_no_metrics(self):
        """Should return None for empty metrics."""
        runner = LmEvalHarnessRunner()
        metrics = {}
        assert runner._extract_primary_score(metrics) is None

    def test_extract_primary_score_harness_key_format(self):
        """lm-eval 0.4.x keys metrics as '<metric>,<filter>' plus an 'alias' entry."""
        runner = LmEvalHarnessRunner()
        metrics = {
            "alias": "arc_easy",
            "acc,none": 0.61,
            "acc_stderr,none": 0.02,
            "acc_norm,none": 0.58,
            "acc_norm_stderr,none": 0.02,
        }
        assert runner._extract_primary_score(metrics) == 0.61

    def test_extract_primary_score_skips_stderr_entries(self):
        """stderr values must never be picked, regardless of dict order."""
        runner = LmEvalHarnessRunner()
        metrics = {
            "alias": "gsm8k",
            "exact_match_stderr,strict-match": 0.01,
            "exact_match,strict-match": 0.42,
            "exact_match,flexible-extract": 0.45,
        }
        assert runner._extract_primary_score(metrics) == 0.42

    @pytest.mark.skip(reason="Requires lm-eval-harness and model download")
    def test_run_evaluation(self):
        """Test actual evaluation run (requires dependencies)."""
        runner = LmEvalHarnessRunner()

        spec = BenchmarkSpec(
            backend="lm-eval-harness",
            task="arc_easy",
            num_fewshot=0,
            limit=10,
        )

        cfg = EvalConfig(
            base_model="EleutherAI/pythia-70m",
            batch_size=4,
            benchmarks=[spec],
        )

        results = runner.run(spec, cfg)

        assert "score" in results
        assert results["task"] == "arc_easy"
        assert results["model"] == "EleutherAI/pythia-70m"
        assert "metrics" in results


FAKE_HARNESS_RESULTS = {
    "results": {
        "arc_easy": {
            "alias": "arc_easy",
            "acc,none": 0.61,
            "acc_stderr,none": 0.02,
            "acc_norm,none": 0.58,
            "acc_norm_stderr,none": 0.02,
        }
    },
    "n-shot": {"arc_easy": 0},
    "n-samples": {"arc_easy": {"original": 2376, "effective": 10}},
    "higher_is_better": {"arc_easy": {"acc": True, "acc_norm": True}},
}


def _install_fake_lm_eval(monkeypatch, captured_kwargs):
    """Register a minimal fake lm_eval package that records simple_evaluate calls."""
    lm_eval_mod = types.ModuleType("lm_eval")
    tasks_mod = types.ModuleType("lm_eval.tasks")
    evaluator_mod = types.ModuleType("lm_eval.evaluator")

    class TaskManager:
        pass

    def get_task_dict(task_name_list, task_manager=None):
        return {name: object() for name in task_name_list}

    def simple_evaluate(**kwargs):
        captured_kwargs.update(kwargs)
        return FAKE_HARNESS_RESULTS

    tasks_mod.TaskManager = TaskManager
    tasks_mod.get_task_dict = get_task_dict
    evaluator_mod.simple_evaluate = simple_evaluate
    lm_eval_mod.tasks = tasks_mod
    lm_eval_mod.evaluator = evaluator_mod

    monkeypatch.setitem(sys.modules, "lm_eval", lm_eval_mod)
    monkeypatch.setitem(sys.modules, "lm_eval.tasks", tasks_mod)
    monkeypatch.setitem(sys.modules, "lm_eval.evaluator", evaluator_mod)


class TestRunWiring:
    """run() must drive simple_evaluate the way lm-eval 0.4.x expects."""

    def _run(self, monkeypatch, spec_kwargs=None, cfg_kwargs=None):
        captured = {}
        _install_fake_lm_eval(monkeypatch, captured)

        spec = BenchmarkSpec(
            backend="lm-eval-harness", task="arc_easy", **(spec_kwargs or {})
        )
        cfg = EvalConfig(
            base_model="EleutherAI/pythia-70m",
            benchmarks=[spec],
            **(cfg_kwargs or {}),
        )
        result = LmEvalHarnessRunner().run(spec, cfg)
        return captured, result

    def test_model_type_and_model_args(self, monkeypatch):
        captured, _ = self._run(
            monkeypatch,
            spec_kwargs={"limit": 10},
            cfg_kwargs={"revision": "step3000", "batch_size": 4},
        )

        # "hf" is the registered model type; the checkpoint goes in model_args.
        assert captured["model"] == "hf"
        assert captured["model_args"]["pretrained"] == "EleutherAI/pythia-70m"
        assert captured["model_args"]["revision"] == "step3000"
        assert captured["model_args"]["dtype"] == "auto"
        assert captured["model_args"]["max_length"] == 4096
        assert captured["tasks"] == ["arc_easy"]
        assert captured["batch_size"] == 4
        assert captured["limit"] == 10
        # Not a simple_evaluate parameter — passing it raises TypeError.
        assert "output_path" not in captured

    def test_unset_fewshot_defers_to_task_default(self, monkeypatch):
        captured, result = self._run(monkeypatch)
        assert captured["num_fewshot"] is None
        # Reported fewshot is the effective value from the harness.
        assert result["num_fewshot"] == 0

    def test_result_shape(self, monkeypatch):
        _, result = self._run(monkeypatch, spec_kwargs={"limit": 10})

        assert result["score"] == 0.61
        assert result["task"] == "arc_easy"
        assert result["model"] == "EleutherAI/pythia-70m"
        assert result["n_samples"] == 10
        assert result["metrics"]["acc,none"] == 0.61
        assert result["details"]["higher_is_better"] == {"acc": True, "acc_norm": True}
