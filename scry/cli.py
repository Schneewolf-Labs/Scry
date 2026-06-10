"""Simple CLI for running evaluations.

v0.1: Basic command-line interface for running single-benchmark evaluations.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from .config import EvalConfig, BenchmarkSpec
from .runners import LmEvalHarnessRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Scry - Evaluate HuggingFace models with benchmarks"
    )
    
    # Model configuration
    parser.add_argument(
        "--model", "--base-model",
        required=True,
        help="HuggingFace model ID or local path"
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Model revision (commit/branch)"
    )
    
    # Benchmark configuration
    parser.add_argument(
        "--task",
        required=True,
        help="lm-eval-harness task name (e.g., 'arc_easy', 'hellaswag', 'truthfulqa_mc2')"
    )
    parser.add_argument(
        "--num-fewshot",
        type=int,
        default=None,
        help="Number of few-shot examples (default: the task's own default)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples to evaluate"
    )
    
    # Evaluation parameters
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for evaluation (default: 8)"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=4096,
        help="Maximum sequence length (default: 4096)"
    )
    
    # Output
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output file for JSON results"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Build config
        spec = BenchmarkSpec(
            backend="lm-eval-harness",
            task=args.task,
            num_fewshot=args.num_fewshot,
            limit=args.limit,
        )

        cfg = EvalConfig(
            base_model=args.model,
            revision=args.revision,
            batch_size=args.batch_size,
            max_length=args.max_length,
            benchmarks=[spec],
        )

        # Run evaluation
        logger.info(f"Evaluating {cfg.base_model} on {spec.task}")
        fewshot_desc = spec.num_fewshot if spec.num_fewshot is not None else "task default"
        logger.info(f"Few-shot: {fewshot_desc}, Limit: {spec.limit}")
        
        runner = LmEvalHarnessRunner()
        results = runner.run(spec, cfg)
        
        # Output results
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results written to {args.output}")
        else:
            print(json.dumps(results, indent=2))
        
        # Exit with appropriate code
        if results.get("score") is None:
            sys.exit(1)
        
    except ImportError as e:
        logger.error(f"Dependency error: {e}")
        sys.exit(2)
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(3)
    except RuntimeError as e:
        logger.error(f"Evaluation error: {e}")
        sys.exit(4)
    except KeyboardInterrupt:
        logger.info("Evaluation cancelled by user")
        sys.exit(130)


if __name__ == "__main__":
    main()
