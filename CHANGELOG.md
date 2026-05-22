# Changelog

All notable changes to Scry will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-05-21

### Added

- **lm-eval-harness runner**: Single-benchmark evaluation via `LmEvalHarnessRunner`
  - Supports all lm-eval-harness tasks (arc_easy, hellaswag, truthfulqa_mc2, etc.)
  - Configurable few-shot learning, sample limits, batch sizes
  - Returns structured JSON results with metrics and metadata
- **CLI entrypoint**: `scry-eval` command for command-line evaluation
  - Specify model, task, and evaluation parameters
  - JSON output to file or stdout
  - Verbose logging option
- **Python API**: Programmatic access to evaluation runner
  - `EvalConfig` and `BenchmarkSpec` Pydantic models
  - `LmEvalHarnessRunner.run()` method for evaluations
  - Easy integration into custom workflows
- **Testing infrastructure**: pytest setup with unit tests for runner

### Changed

- Updated development status from "1 - Planning" to "4 - Beta"

### Known Limitations

- No job queue or WebSocket progress (planned for v0.2)
- No paired-evaluation / lineage comparison (planned for v0.3)
- No VLMEvalKit or BFCL support (planned for v0.4)
- No W&B or HuggingFace model card integration (planned for v0.5-v0.6)

[0.1.0]: https://github.com/Schneewolf-Labs/Scry/releases/tag/v0.1.0
