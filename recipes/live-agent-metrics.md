# Recipe: live agent metrics (#10 / RM-184)

Multi-phase cold → tool → resume timings on the OpenAI-compatible client path (#3 / #12).

## Prerequisites

- Engine up: Ollama (`11434`) or llama-server (`8080`)
- `configs/engine.example.env` (or `engine.local.env`)

```bash
export BKL_BASE_URL=http://127.0.0.1:11434/v1
export BKL_MODEL=granite4.1:8b
# optional: export BKL_ENGINE_VERSION="$(ollama --version)"
```

## Run

```bash
export BKL_ENGINE=ollama   # or llama-server — recorded in results JSON
python3 harness/agent_loop/run_live_metrics.py --out results/ --loops 1
# more samples: cycle 0 = cold; later cycles are warm (not labeled cold)
python3 harness/agent_loop/run_live_metrics.py --out results/ --loops 3 --label live-tool-loop
# fail hard if model ignores TOOL protocol:
python3 harness/agent_loop/run_live_metrics.py --out results/ --require-tool-protocol
python3 harness/report/aggregate.py --results results/
# fixtures for MATRIX may be copied under docs/results/ (sanitized, intentional)
```

## What is measured

| Metric | Phase |
|--------|--------|
| `ttft_cold_ms` | First turn: system+tools + user |
| `ttft_resume_ms` | After simulated tool result append |
| `tpot_ms` | Derived from stream wall − TTFT / tokens |
| `tool_loop_wall_ms` | Full cold→resume cycle |
| `vram_peak_mb` | max(before, after) samples |

Tool execution is **simulated** (fixed `TOOL_RESULT` text) so the harness isolates model prefill/decode, not shell latency.

## Client path (#3)

Shared client: `harness/serve/openai_compat.py`  
Smoke: `harness/serve/smoke_openai.py`  
Config: `configs/engine.example.env`

## Matrix

Add/update rows in [docs/MATRIX.md](../docs/MATRIX.md) from the written `results/*.json`.
