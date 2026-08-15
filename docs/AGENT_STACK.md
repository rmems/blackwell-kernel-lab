# Agent stack — this host

Primary path for **running local agents** on the RTX 5080. Training / fine-tunes belong in **`agoge-forger`**.

Kernel campaign overview: [KERNELS.md](KERNELS.md) (L1 measure engine CUDA paths first).

## Recommended defaults

| Layer | Choice | Why |
|-------|--------|-----|
| Inference (**primary**) | **Ollama** → `http://127.0.0.1:11434/v1` | On host; smoke verified; OpenAI-compatible |
| Inference (alt) | **llama-server** (Homebrew) | Better FA/graph flag control for #16 ablations |
| Inference (optional M2) | vLLM / SGLang / imp | Only if sm_120 install is clean |
| Agent | Coding / tool agent via **OpenAI-compatible** HTTP | Local agent loops |
| Measure | `smoke_openai.py` + `run_live_metrics.py` → `results/` | Smoke TTFT + cold/resume tool-loop |
| Client | `harness/serve/openai_compat.py` | OpenAI-compatible streaming (#3) |

## Primary engine decision (ShipOfTheseus)

| Field | Value |
|-------|--------|
| Engine | **Ollama 0.30.x** |
| Endpoint | `http://127.0.0.1:11434/v1` |
| Smoke model | `granite4.1:8b` (present locally; ~5–10 GB class) |
| Smoke | `python3 harness/serve/smoke_openai.py --stream` → content `kernel-smoke-ok` |
| Headroom | `ollama stop <model>` after runs; target ≥2 GB free when idle multi-tasking |

**Note:** While a model is loaded, free VRAM can drop hard (observed ~1 GB free mid-run on `granite4.1:8b`). That **violates** the ≥2 GB free multi-task headroom rule while the model is resident. Prefer smaller models for concurrent desktop use, or `ollama stop` between agent sessions. For strict measurement, pass `--min-free-vram-mb 2048` only when free VRAM is part of the SLO.

**Root cause (measured 2026-08-15, #13):** that ~1 GB free came from Ollama loading `granite4.1:8b` at its
advertised **131072** context — a ~27 GB allocation, 53% offloaded to CPU. Pinning `num_ctx 8192` in a derived
tag puts the same weights **100% on GPU**: 7131 MiB free, decode 45.1 → 7.35 ms/token.

```bash
printf 'FROM granite4.1:8b\nPARAMETER num_ctx 8192\n' > /tmp/Modelfile.ctx8k
ollama create granite4.1:8b-ctx8k -f /tmp/Modelfile.ctx8k   # reuses blobs, no download
ollama ps   # PROCESSOR must read 100% GPU, not 53%/47% CPU/GPU
```

`run_live_metrics.py` records this as `model.residency` from `/api/ps`. Numbers from a partially resident model
are CPU numbers — see [MODELS.md](MODELS.md).

## Engines on ShipOfTheseus

### Ollama (primary)

```bash
ollama --version
ollama list
export BKL_BASE_URL=http://127.0.0.1:11434/v1
export BKL_MODEL=granite4.1:8b
python3 harness/serve/smoke_openai.py --out results/ --stream --label cell-A-baseline
ollama stop "$BKL_MODEL"
```

### llama.cpp (`llama-server`) — preferred for kernel flag ablations

ShipOfTheseus measured path (**#16 / RM-470**, 2026-08-12):

| Field | Value |
|-------|--------|
| Binary | Homebrew `llama-server` **9190** (`b64739ea3`) |
| Model | `gemma-4-E2B-it-UD-Q4_K_XL.gguf` (~3.0 GiB GGUF, ~4.6B params class) |
| Endpoint | `http://127.0.0.1:8080/v1` |
| GPU layers | `-ngl 99` |
| Context | `-c 4096` |
| Flash Attention | **`-fa off`** (cell A) vs **`-fa on`** (cells B/C) — CLI works on this build |
| CUDA graphs | **No CLI toggle** on 9190 — engine/CUDA default only. Cell C = FA on + graphs=default (honest fallback, not a fake off/on win) |
| Quant D | No second MXFP4/NVFP4 sibling for this GGUF on host yet — **cell D deferred** |

```bash
llama-server --version   # expect 9190 on this host
# Cell A
llama-server -m /path/to/gemma-4-E2B-it-UD-Q4_K_XL.gguf -ngl 99 -c 4096 --port 8080 -fa off
# Cell B/C
llama-server -m /path/to/gemma-4-E2B-it-UD-Q4_K_XL.gguf -ngl 99 -c 4096 --port 8080 -fa on

export BKL_BASE_URL=http://127.0.0.1:8080/v1
export BKL_MODEL=gemma-4-E2B-it-UD-Q4_K_XL.gguf
# Or full matrix:
# MODEL=/path/to.gguf bash harness/serve/run_l1_ablation.sh
```

JSON snapshots: `docs/results/l1-ablation-2026-08-12/`.

## Agent loop shape

Local agents are **not** long chatbots:

1. **Cold prefill** — system prompt + tool schemas  
2. **Resume prefill** — tool results into cached context  
3. **Short structured decode** — tool calls / JSON  

Optimize for **tool-loop latency** and **TPOT stability**, not only peak batch tokens/s.

Named versions of these shapes live in `harness/agent_loop/profiles.py` (`live_agent_tool_loop`,
`live_coding_tool`, `live_plan_exec`) and are shared by the live harness and the #16 smoke — see
[WORKLOADS.md](WORKLOADS.md).

## Efficiency levers (measure on this box)

| Lever | Expectation |
|-------|-------------|
| Quant (Q4 / NVFP4 / MXFP4) | Fit + speed under 16 GB |
| Flash Attention | Prefill + KV memory |
| CUDA graphs | Lower decode launch overhead |
| Prefix / session cache | Shared tools + multi-turn |
| Speculative decode | Often weak under constrained tool JSON — measure |
| Multi-agent isolation | Process limits; later Green Context notes |

## Out of scope here

- Full training loops → **`agoge-forger`**  
- Shipping a second CUDA kernel library → myelin if needed  
