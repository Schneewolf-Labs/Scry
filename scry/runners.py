"""Eval-backend runner stubs.

Each backend (lm-eval-harness, VLMEvalKit, BFCL, custom batteries) will get a
concrete runner here. For the foundation release we only sketch the interface
the FastAPI layer will call into — so the API contract can be designed
without waiting for the harnesses to be integrated.

v0.1: LmEvalHarnessRunner is now implemented for single-benchmark evaluation.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional
import logging

from .config import BenchmarkSpec, EvalConfig

logger = logging.getLogger(__name__)


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
        progress_cb: Optional[Callable[[int, int, str], None]] = None,
    ) -> dict[str, Any]:
        """Run one benchmark task. Return a results dict whose schema is the
        intersection across backends (at least: `score`, `task`, `n_samples`,
        plus arbitrary backend-specific extras under `details`).

        `progress_cb` is an optional `Callable[[int, int, str], None]` that
        receives (current_sample, total_samples, message); the FastAPI layer
        will hand in a callback that pushes WebSocket updates.
        """


# ============================================================================
# lm-eval-harness runner (v0.1)
# ============================================================================

class LmEvalHarnessRunner(BaseRunner):
    """Wraps EleutherAI/lm-evaluation-harness via its Python API.

    Implements v0.1: single-benchmark evaluation on one HF model.
    
    Usage:
        runner = LmEvalHarnessRunner()
        results = runner.run(spec, cfg)
        
    Will install lm-eval-harness as an extra (`pip install scry[harness]`)
    rather than pinning a hard dependency, so users who only want VLM eval
    don't pay the cost.
    """
    backend_name = "lm-eval-harness"

    def can_handle(self, spec: BenchmarkSpec) -> bool:
        return spec.backend == "lm-eval-harness"

    def run(self, spec: BenchmarkSpec, cfg: EvalConfig, progress_cb=None) -> dict[str, Any]:
        """Run a single lm-eval-harness benchmark.
        
        Args:
            spec: Benchmark specification (task name, fewshot, limit)
            cfg: Evaluation config (model, batch size, etc.)
            progress_cb: Optional callback for progress updates
            
        Returns:
            Results dict with score, task metadata, and details
            
        Raises:
            ImportError: If lm-eval-harness is not installed
            ValueError: If the task is not found
        """
        # Import lm-eval-harness components (will raise ImportError if not installed)
        try:
            from lm_eval.tasks import TaskManager, get_task_dict
            from lm_eval.evaluator import evaluator, simple_evaluate
            from lm_eval.models import get_model
        except ImportError as e:
            raise ImportError(
                "lm-eval-harness is not installed. Install with: pip install scry[harness]"
            ) from e
        
        logger.info(f"Running lm-eval-harness task '{spec.task}' on model '{cfg.base_model}'")
        
        # Build the task list (lm-eval-harness uses comma-separated task names)
        task_list = spec.task
        
        # Get task manager and resolve task names
        task_manager = TaskManager()
        try:
            task_dict = get_task_dict([task_list], task_manager)
        except Exception as e:
            raise ValueError(f"Task '{spec.task}' not found in lm-eval-harness: {e}") from e
        
        # Determine fewshot and limit
        num_fewshot = spec.num_fewshot if spec.num_fewshot is not None else 0
        limit = spec.limit if spec.limit is not None else None
        
        # Run evaluation using simple_evaluate (higher-level API)
        try:
            results = simple_evaluate(
                model=cfg.base_model,
                tasks=[task_list],
                num_fewshot=num_fewshot,
                limit=limit,
                batch_size=cfg.batch_size,
                output_path=None,  # We'll handle output ourselves
                verbosity="INFO",
            )
        except Exception as e:
            raise RuntimeError(f"Evaluation failed: {e}") from e
        
        # Extract results for the task
        task_results = results["results"].get(spec.task, {})
        
        # Build the result dict
        result = {
            "task": spec.task,
            "backend": self.backend_name,
            "model": cfg.base_model,
            "revision": cfg.revision,
            "num_fewshot": num_fewshot,
            "limit": limit,
            "batch_size": cfg.batch_size,
            "metrics": task_results,
            "details": {
                "aggregation": task_results.get("aggregation"),
                "higher_is_better": task_results.get("higher_is_better"),
            },
        }
        
        # Extract the primary score (usually the first metric or "acc")
        primary_score = self._extract_primary_score(task_results)
        result["score"] = primary_score
        
        logger.info(f"Completed evaluation: score={primary_score}")
        
        return result
    
    def _extract_primary_score(self, metrics: dict) -> Optional[float]:
        """Extract the primary score from lm-eval-harness metrics.
        
        lm-eval-harness returns multiple metrics; we extract the most relevant one:
        - If "acc" exists, use it (common for classification)
        - Otherwise, use the first metric that looks like a score
        """
        # Priority order for primary metrics
        priority_metrics = ["acc", "accuracy", "f1", "bleu", "perplexity"]
        
        for metric in priority_metrics:
            if metric in metrics and isinstance(metrics[metric], (int, float)):
                return float(metrics[metric])
        
        # Fallback: return first numeric metric
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not key.startswith("stderr"):
                return float(value)
        
        return None


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
