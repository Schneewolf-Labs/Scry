"""Pydantic config schemas for Scry eval requests.

These define the *shape* of what POST /eval will accept — the runner that
actually consumes them lives in `scry/runners.py` (stub) and `scry/app.py`
(stub). The schema can settle here while the runner is being built so
contributors / API consumers have a stable target.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


# Benchmarks fall in three rough families. We name them with a leading prefix
# so the runner can dispatch on the family without string-matching the full
# task list. Concrete task names per family are loaded from each backend's
# registry at request time.
BackendName = Literal["lm-eval-harness", "vlm-eval-kit", "bfcl", "custom"]


class BenchmarkSpec(BaseModel):
    """One benchmark task to run.

    Examples:
        BenchmarkSpec(backend="lm-eval-harness", task="ifeval")
        BenchmarkSpec(backend="bfcl", task="single_function_call")
        BenchmarkSpec(backend="vlm-eval-kit", task="mme")
    """

    backend: BackendName = Field(
        ..., description="Eval backend that owns this task"
    )
    task: str = Field(
        ...,
        description="Backend-specific task name (e.g. 'ifeval', 'mmlu-pro', "
                    "'bfcl_v3_single', 'mme', 'mmstar')"
    )
    num_fewshot: Optional[int] = Field(
        None, ge=0, le=32,
        description="Backend's fewshot count if applicable (lm-eval-harness)"
    )
    limit: Optional[int] = Field(
        None, ge=1,
        description="Cap samples evaluated (useful for smoke runs)"
    )


class EvalConfig(BaseModel):
    """Top-level POST /eval body.

    A single config can drive multiple benchmarks against a primary model,
    optionally comparing against a list of baseline models (paired-eval mode).
    """

    # Model under test
    base_model: str = Field(
        ..., description="HuggingFace repo id or local path of the model to evaluate"
    )
    revision: Optional[str] = Field(
        None, description="HF revision (commit / branch) for reproducibility"
    )

    # Paired-evaluation: produce delta reports against these baselines
    compare_to: list[str] = Field(
        default_factory=list,
        description=(
            "Other model ids to run the *same* benchmarks against, for "
            "side-by-side delta reports (the lineage comparison feature)"
        ),
    )

    # Benchmarks
    benchmarks: list[BenchmarkSpec] = Field(
        ..., min_length=1,
        description="One or more benchmark tasks to run"
    )

    # Eval-time knobs
    batch_size: int = Field(8, ge=1, le=128)
    max_length: int = Field(4096, ge=128, le=131072)
    dtype: Literal["auto", "bfloat16", "float16", "float32"] = Field("auto")

    # Outputs
    output_name: Optional[str] = Field(
        None,
        description="Job-readable name for results dir; defaults to a slug of base_model"
    )
    push_results_to_hub: bool = Field(
        False,
        description="Open a PR against base_model's HF repo appending a results table to the model card"
    )
    use_wandb: bool = Field(False)
    wandb_project: Optional[str] = Field(None)
    wandb_run_name: Optional[str] = Field(None)
    hf_token: Optional[str] = Field(None, description="Required for private models / hub PRs")
