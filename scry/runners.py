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
            RuntimeError: If the evaluation itself fails
        """
        # Import lm-eval-harness components (will raise ImportError if not installed)
        try:
            from lm_eval.evaluator import simple_evaluate
            from lm_eval.tasks import TaskManager, get_task_dict
        except ImportError as e:
            raise ImportError(
                "lm-eval-harness is not installed. Install with: pip install scry[harness]"
            ) from e

        logger.info(f"Running lm-eval-harness task '{spec.task}' on model '{cfg.base_model}'")

        # Validate the task name up front so an unknown task surfaces as a
        # ValueError instead of failing deep inside the harness.
        task_manager = TaskManager()
        try:
            get_task_dict([spec.task], task_manager)
        except Exception as e:
            raise ValueError(f"Task '{spec.task}' not found in lm-eval-harness: {e}") from e

        # The harness takes the model *type* in `model` ("hf" = HuggingFace
        # transformers); the checkpoint and its loading knobs go in `model_args`.
        model_args: dict[str, Any] = {
            "pretrained": cfg.base_model,
            "dtype": cfg.dtype,
            "max_length": cfg.max_length,
        }
        if cfg.revision is not None:
            model_args["revision"] = cfg.revision

        # num_fewshot=None lets the task's own default apply; an explicit
        # value (including 0) overrides it.
        try:
            results = simple_evaluate(
                model="hf",
                model_args=model_args,
                tasks=[spec.task],
                num_fewshot=spec.num_fewshot,
                limit=spec.limit,
                batch_size=cfg.batch_size,
                task_manager=task_manager,
            )
        except Exception as e:
            raise RuntimeError(f"Evaluation failed: {e}") from e

        # Extract results for the task
        task_results = results["results"].get(spec.task, {})
        n_samples = results.get("n-samples", {}).get(spec.task, {})

        # Build the result dict
        result = {
            "task": spec.task,
            "backend": self.backend_name,
            "model": cfg.base_model,
            "revision": cfg.revision,
            "num_fewshot": results.get("n-shot", {}).get(spec.task, spec.num_fewshot),
            "limit": spec.limit,
            "batch_size": cfg.batch_size,
            "n_samples": n_samples.get("effective"),
            "metrics": task_results,
            "details": {
                "n_samples": n_samples,
                "higher_is_better": results.get("higher_is_better", {}).get(spec.task),
            },
        }

        # Extract the primary score (usually the first metric or "acc")
        primary_score = self._extract_primary_score(task_results)
        result["score"] = primary_score

        logger.info(f"Completed evaluation: score={primary_score}")

        return result
    
    def _extract_primary_score(self, metrics: dict) -> Optional[float]:
        """Extract the primary score from lm-eval-harness metrics.

        lm-eval-harness 0.4.x keys metrics as "<metric>,<filter>" (e.g.
        "acc,none", "acc_stderr,none") alongside a non-numeric "alias" entry.
        We extract the most relevant one:
        - If "acc" exists, use it (common for classification)
        - Otherwise, use the first metric that looks like a score
        """
        # Priority order for primary metrics
        priority_metrics = ["acc", "accuracy", "exact_match", "acc_norm", "f1", "bleu", "perplexity"]

        # Index numeric metrics by bare metric name, dropping stderr entries.
        # First filter wins when a metric is reported under several filters.
        scores: dict[str, float] = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and "stderr" not in key:
                scores.setdefault(key.split(",", 1)[0], float(value))

        for metric in priority_metrics:
            if metric in scores:
                return scores[metric]

        # Fallback: return first numeric metric
        return next(iter(scores.values()), None)


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
