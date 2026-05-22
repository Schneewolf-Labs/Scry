#!/usr/bin/env python3
"""Example: Evaluate a model on a benchmark.

This script demonstrates how to use Scry's Python API to run a single-benchmark
evaluation on a HuggingFace model.
"""
import json
import logging
from scry import EvalConfig, BenchmarkSpec, LmEvalHarnessRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    # Configure the evaluation
    cfg = EvalConfig(
        base_model="EleutherAI/pythia-70m",  # Small model for quick demo
        batch_size=4,
        max_length=2048,
    )
    
    # Specify the benchmark task
    spec = BenchmarkSpec(
        backend="lm-eval-harness",
        task="arc_easy",  # AI2 Reasoning Challenge (easy subset)
        num_fewshot=0,    # Zero-shot evaluation
        limit=20,         # Only evaluate 20 samples (for speed)
    )
    
    # Run the evaluation
    logger.info(f"Evaluating {cfg.base_model} on {spec.task}")
    logger.info(f"Configuration: {spec.num_fewshot}-shot, {spec.limit} samples")
    
    runner = LmEvalHarnessRunner()
    results = runner.run(spec, cfg)
    
    # Display results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Model: {results['model']}")
    print(f"Task: {results['task']}")
    print(f"Primary Score: {results['score']:.4f}" if results['score'] else "Primary Score: N/A")
    print(f"\nAll Metrics:")
    for key, value in results['metrics'].items():
        if not key.endswith('_stderr') and isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    
    # Save results to JSON
    output_file = "evaluation_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
