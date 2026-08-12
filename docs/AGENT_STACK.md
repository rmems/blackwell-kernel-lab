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
| Measure | `harness/serve/smoke_openai.py` → `results/` | TTFT/wall/VRAM |

## Primary engine decision (ShipOfTheseus)

| Field | Value |
|-------|--------|
| Engine | **Ollama 0.30.x** |
| Endpoint | `http://127.0.0.1:11434/v1` |
| Smoke model | `granite4.1:8b` (present locally; ~5–10 GB class) |
| Smoke | `python3 harness/serve/smoke_openai.py --stream` → content `kernel-smoke-ok` |
| Headroom | `ollama stop <model>` after runs; target ≥2 GB free when idle multi-tasking |

**Note:** While a model is loaded, free VRAM can drop hard (observed ~1 GB free mid-run). Always stop models when done.

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

```bash
llama-server --version
# Example (adjust model path):
# llama-server -m /path/to/model.gguf -ngl 99 -c 8192 --port 8080
# Use build docs for flash-attn + CUDA graph flags on your revision
```

Document exact flags that work on **sm_120** in `results/*.json` notes and here when known.

## Agent loop shape

Local agents are **not** long chatbots:

1. **Cold prefill** — system prompt + tool schemas  
2. **Resume prefill** — tool results into cached context  
3. **Short structured decode** — tool calls / JSON  

Optimize for **tool-loop latency** and **TPOT stability**, not only peak batch tokens/s.

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
