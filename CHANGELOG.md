# Changelog

All notable changes to Scry will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-06-12

### Added

- **FastAPI server** (`scry.app.create_app`, `scry-server` entrypoint):
  - `POST /eval` accepts a full `EvalConfig` — **multiple benchmarks per
    request** — and returns `202` with a `job_id`
  - `GET /jobs` (history, newest first), `GET /jobs/{id}`,
    `POST /jobs/{id}/stop`, `GET /results/{id}`, `GET /health`
  - `WS /ws/jobs/{id}` streams live events: a `snapshot` on connect, then
    `benchmark_started` / `progress` / `benchmark_completed` and a terminal
    `status` event
- **Job queue** (`scry.jobs`): single worker thread so one eval owns the GPU
  at a time; sqlite-backed history (`./data/eval_jobs.db`, `--db` to
  override); crash recovery on restart (queued jobs re-enqueued, interrupted
  jobs marked failed)
- **Stop/cancel semantics**: stopping a queued job cancels it outright;
  stopping a running job halts it after the current benchmark, keeping
  partial results
- **Runner dispatch**: `default_runners()` / `get_runner()` pick the backend
  per `BenchmarkSpec`; a failing benchmark fails the job but keeps the
  results of benchmarks that already completed
- `hf_token` is masked in every API response and job serialization

### Changed

- `scry[api]` extra is now live (FastAPI + uvicorn + websockets) and installs
  the `scry-server` console script
- CI installs the `api` extra so the FastAPI tests run (they skip themselves
  when FastAPI isn't installed)

## [0.1.1] - 2026-06-10

### Fixed

- **CLI and example crashed on startup**: `scry-eval` and `examples/evaluate_model.py`
  built `EvalConfig` without the required `benchmarks` field, so every invocation
  failed validation before reaching the runner. The README Python example had the
  same bug.
- **lm-eval-harness integration never worked**:
  - Imports referenced names that don't exist (`lm_eval.evaluator.evaluator`,
    `lm_eval.models.get_model`), so an installed harness was misreported as
    "not installed".
  - `simple_evaluate` was called with the HF repo id as `model`; the harness
    expects the model *type* (`"hf"`) there, with the checkpoint in
    `model_args["pretrained"]`.
  - `output_path` is not a `simple_evaluate` parameter and raised `TypeError`.
  - `EvalConfig.revision`, `dtype`, and `max_length` were silently ignored;
    they are now passed through to the model loader.
- **Primary-score extraction** now understands lm-eval 0.4.x `"<metric>,<filter>"`
  keys (e.g. `"acc,none"`), never picks `*_stderr` values, and also recognizes
  `exact_match`/`acc_norm`.
- Results now include `n_samples` (per the `BaseRunner` contract) and a correctly
  sourced `higher_is_better`.

### Changed

- Leaving `num_fewshot` unset now uses the task's own default instead of forcing
  zero-shot; pass `--num-fewshot 0` explicitly for zero-shot.

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

[0.2.0]: https://github.com/Schneewolf-Labs/Scry/releases/tag/v0.2.0
[0.1.1]: https://github.com/Schneewolf-Labs/Scry/releases/tag/v0.1.1
[0.1.0]: https://github.com/Schneewolf-Labs/Scry/releases/tag/v0.1.0
