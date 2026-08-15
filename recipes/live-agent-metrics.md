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

## Pin the context first (Ollama)

Ollama loads at the model's advertised context (131072 for `granite4.1:8b`) and **silently offloads to CPU**
on a 16 GB card. Measured 2026-08-15: 47% resident, 6.1× slower decode ([MODELS.md](../docs/MODELS.md)).

```bash
printf 'FROM granite4.1:8b\nPARAMETER num_ctx 8192\n' > /tmp/Modelfile.ctx8k
ollama create granite4.1:8b-ctx8k -f /tmp/Modelfile.ctx8k   # reuses blobs, no download
export BKL_MODEL=granite4.1:8b-ctx8k
ollama ps   # PROCESSOR must read 100% GPU
```

llama-server: pass `-c 8192 -ngl 99` instead; there is no residency probe on that path.

## Run

```bash
export BKL_ENGINE=ollama   # or llama-server — recorded in results JSON
python3 harness/agent_loop/run_live_metrics.py --out results/ --loops 1
# more samples: cycle 0 = cold; later cycles are warm (not labeled cold)
python3 harness/agent_loop/run_live_metrics.py --out results/ --loops 3 --label live-tool-loop
# named workload profiles (docs/WORKLOADS.md):
python3 harness/agent_loop/run_live_metrics.py --out results/ --profile live_coding_tool --loops 3
python3 harness/agent_loop/run_live_metrics.py --out results/ --profile live_plan_exec --loops 2
# fail hard if model ignores TOOL protocol:
python3 harness/agent_loop/run_live_metrics.py --out results/ --require-tool-protocol
python3 harness/report/aggregate.py --results results/
# fixtures for MATRIX may be copied under docs/results/ (sanitized, intentional)
```

Profiles: `live_agent_tool_loop` (default), `live_coding_tool`, `live_plan_exec` — text frozen in
`harness/agent_loop/profiles.py`. Each profile sets its own `max_tokens`; `--max-tokens` overrides.
Thinking models need a much larger budget (measured: 96 returned no content, 512 did) and their
TTFT/TPOT are not comparable to non-thinking models.

## What is measured

| Metric | Phase |
|--------|--------|
| `ttft_cold_ms` | First turn: system+tools + user |
| `ttft_resume_ms` | After simulated tool result append |
| `tpot_ms` | Derived from stream wall − TTFT / tokens |
| `tool_loop_wall_ms` | Full cold→resume cycle |
| `vram_peak_mb` | max(before, after) samples |
| `metrics.phases[]` | Per-cycle breakdown: cold prefill, resume prefill, decode ms, prompt/completion tokens |
| `model.residency` | Ollama `/api/ps`: `size_vram / size`, `fully_gpu_resident` (null on other engines) |

Tool execution is **simulated** (fixed `TOOL_RESULT` text) so the harness isolates model prefill/decode, not shell latency.

## Client path (#3)

Shared client: `harness/serve/openai_compat.py`  
Smoke: `harness/serve/smoke_openai.py`  
Config: `configs/engine.example.env`

## Matrix

Add/update rows in [docs/MATRIX.md](../docs/MATRIX.md) from the written `results/*.json`.
