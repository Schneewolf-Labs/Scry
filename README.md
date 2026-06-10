# Scry — Schneewolf Labs

> *To **scry**: to gaze into a crystal, mirror, or model and see what it can truly do.*

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-v0.1%20MVP-green.svg)]()

The eval-orchestrator counterpart to [Merlina](https://github.com/Schneewolf-Labs/Merlina).
Merlina trains the model; **Scry judges it**.

> ✅ **v0.1 MVP: Single-benchmark evaluation is now working!** Run lm-eval-harness
> benchmarks on HuggingFace models via Python API or CLI. Job queue, WebSocket
> progress, and paired-evaluation coming in v0.2-v0.3. See [Roadmap](#roadmap).

## The vision

Solo and small labs ship checkpoints in *lineage* — A1 → A2 → A3 → Artemis —
and they think in *deltas*, not absolutes. The current eval ecosystem doesn't
serve that shape well:

- HuggingFace's Open LLM Leaderboard is a public submission queue with a fixed
  benchmark menu; you can't ad-hoc compare two of your own checkpoints.
- `lm-evaluation-harness`, `VLMEvalKit`, and `OpenCompass` are powerful but
  CLI-only — no job queue, no per-benchmark progress, no result UI, no
  side-by-side comparison machinery.
- Commercial LLM-monitoring platforms (Galileo, Patronus, HumanLoop) are for
  production observability, not "I have an HF checkpoint, please run BFCL."

Scry fills the gap with a Merlina-shaped wrapper:

```
POST /eval
  base_model:   "schneewolflabs/A2"
  benchmarks:   ["bfcl", "ifeval", "mmlu-pro", "vlm_bench"]
  compare_to:   ["schneewolflabs/A1"]      # paired-eval: produce deltas
  wandb_project: "schneewolf-labs-evals"
  push_results_to_hub: true                 # auto-append leaderboard rows to the model card
```

Single-GPU job queue. WebSocket progress per benchmark. End-state: a results
JSON, a wandb run, and (optionally) a PR to the HF model repo appending a
results table to the card.

## Quickstart (v0.1)

Install with lm-eval-harness support:

```bash
pip install "scry[harness]"
```

Or from source:

```bash
git clone https://github.com/Schneewolf-Labs/Scry.git
cd Scry
pip install -e ".[harness]"
```

Run a benchmark via CLI:

```bash
scry-eval --model meta-llama/Llama-2-7b --task arc_easy --num-fewshot 5 --limit 50
```

Or use the Python API:

```python
from scry import EvalConfig, BenchmarkSpec, LmEvalHarnessRunner

spec = BenchmarkSpec(backend="lm-eval-harness", task="arc_easy", num_fewshot=5)
cfg = EvalConfig(base_model="meta-llama/Llama-2-7b", batch_size=8, benchmarks=[spec])

runner = LmEvalHarnessRunner()
results = runner.run(spec, cfg)

print(f"Score: {results['score']}")
```

The `/eval` API endpoint, job queue, and WebSocket progress land in v0.2.

## What makes Scry different

| Feature | lm-eval-harness CLI | HF Leaderboard | **Scry** |
|---|---|---|---|
| Run on arbitrary HF checkpoint | ✓ | submission queue | ✓ |
| Job queue / WebSocket progress | ✗ | ✗ | ✓ |
| Per-benchmark progress bars | ✗ | ✗ | ✓ |
| **Paired/lineage comparison** | ✗ | ✗ | ✓ (the killer feature) |
| Auto-PR results to HF model card | ✗ | ✗ | ✓ |
| VLM benchmarks (VLMEvalKit) | partial | ✗ | ✓ |
| Tool-calling (BFCL, τ-bench) | manual | ✗ | ✓ |
| Wandb integration | manual | ✗ | ✓ |
| Local + cloud (Modal/RunPod) backends | ✗ | n/a | planned |

The **paired-evaluation** mode is the headline. Solo labs care about
"did A2 actually beat A1 at tool calling?" more than "what's A2's BFCL score
in absolute terms?" Scry treats lineage comparisons as first-class.

## Architecture (planned)

```
                 ┌────────────┐
   POST /eval -> │ FastAPI    │ <-> ./data/eval_jobs.db (sqlite)
                 │ + JobQueue │
                 └─────┬──────┘
                       │
            ┌──────────▼──────────────┐
            │  Eval runner            │   single-GPU, paged from queue
            │  - lm-eval-harness      │
            │  - VLMEvalKit           │
            │  - BFCL submodule       │
            │  - custom batteries     │
            └──────────┬──────────────┘
                       │
            ┌──────────▼──────────────┐
            │  Result aggregator      │
            │  - JSON + markdown      │
            │  - paired diffs         │
            │  - W&B push             │
            │  - HF model-card PR     │
            └─────────────────────────┘
```

Implementation borrows freely from Merlina's existing infrastructure:
`job_manager.py`, `job_queue.py`, `websocket_manager.py`, FastAPI + Pydantic
config patterns are battle-tested there and map cleanly.

## Install

Stable release:

```bash
pip install scry
```

With lm-eval-harness backend:

```bash
pip install "scry[harness]"
```

From source (development):

```bash
git clone https://github.com/Schneewolf-Labs/Scry.git
cd Scry
pip install -e ".[harness,test]"
```

The CLI entrypoint `scry-eval` is installed with the package.

## Roadmap

- [x] **v0.1** — Single-benchmark MVP: `LmEvalHarnessRunner` runs lm-eval-harness
  on one HF model and one task, returns JSON results. CLI entrypoint `scry-eval`
  for command-line usage. No queue, no WebSocket, no comparison.
- [ ] **v0.2** — Job queue + WebSocket progress; multi-benchmark per request.
- [ ] **v0.3** — Paired-evaluation (`compare_to: [...]`) producing delta
  reports.
- [ ] **v0.4** — VLMEvalKit + BFCL integration for the Artemis lineage.
- [ ] **v0.5** — HF model-card auto-PR with results table.
- [ ] **v0.6** — Wandb integration mirroring Merlina's tracker.
- [ ] **v0.7** — Cloud backend (Modal / RunPod / Together) for users without
  local GPU.
- [ ] **v1.0** — UI (similar to Merlina's frontend), public release.

## Why "Scry"

Because divination and oracle metaphors are the natural counterpart to
Merlina's wizardry-training framing, and "Scry" specifically means *to gaze
into a model and see what it can truly do*. Same magical-naming continuity as
Merlina → ArtemisVLM → Scry.

## License

Apache 2.0 — see [LICENSE](LICENSE).
