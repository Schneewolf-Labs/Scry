"""Tests for Scry runners.

v0.1: Tests for lm-eval-harness runner.
"""
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
    
    @pytest.mark.skip(reason="Requires lm-eval-harness and model download")
    def test_run_evaluation(self):
        """Test actual evaluation run (requires dependencies)."""
        runner = LmEvalHarnessRunner()
        
        cfg = EvalConfig(
            base_model="EleutherAI/pythia-70m",
            batch_size=4,
        )
        
        spec = BenchmarkSpec(
            backend="lm-eval-harness",
            task="arc_easy",
            num_fewshot=0,
            limit=10,
        )
        
        results = runner.run(spec, cfg)
        
        assert "score" in results
        assert results["task"] == "arc_easy"
        assert results["model"] == "EleutherAI/pythia-70m"
        assert "metrics" in results
