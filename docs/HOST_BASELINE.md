# Host baseline — local agents on RTX 5080

**Host:** ShipOfTheseus (personal workstation)  
**Refresh:** 2026-08-12T09:50Z (live `nvidia-smi` / `nvcc` on this machine)

## GPU

| Item | Value |
|------|--------|
| GPU | NVIDIA GeForce **RTX 5080** |
| VRAM | **16303 MiB** (~16 GB) |
| Compute capability | **12.0** (`sm_120`, Blackwell consumer) |
| Driver | **610.43.03** |
| Persistence | **Enabled** (`Persistence-M On`) |

```bash
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,compute_cap,driver_version,persistence_mode --format=csv
```

## CUDA toolkit

| Item | Value |
|------|--------|
| Toolkit | **CUDA 13.3** (`/usr/local/cuda`) |
| `nvcc` | **13.3.73** (V13.3.73) |

```bash
nvcc --version
```

## Idle vs loaded VRAM (measured 2026-08-12)

| State | Used MiB | Free MiB | Notes |
|-------|----------|----------|-------|
| **Idle-ish** (no chat model resident; Ollama daemon may still hold a small footprint) | ~**2068** | ~**13811** | After `ollama stop` on chat models |
| **Loaded** `granite4.1:8b` (smoke class) | ~**14261** | ~**1617** | **Violates ≥2 GB free** while model is resident |
| Embedding model left loaded (`qwen3-embedding:8b` observed) | ~**13967** | ~**1911** | Same class of headroom pressure |

**Rules that follow from the numbers:**

1. Prefer **`ollama stop <model>`** between agent sessions when multi-tasking / desktop share.  
2. For strict free-VRAM SLO in smoke: `--min-free-vram-mb 2048` only when the model will not stay loaded.  
3. Smaller models or stop-between-runs are required to meet the ≥2 GB free multi-task rule **while** a 5–10 GB-class model is hot.

## 16 GB agent VRAM rules

Interactive local agents share one card for **weights + KV cache + runtime + desktop**.

| Rule | Guidance |
|------|----------|
| Headroom | Leave **≥2 GB** free after model load for KV + tools + display |
| Model class | Prefer **8–14B dense** or **~20B MoE** at Q4 / NVFP4 / MXFP4 |
| Context | Prefer **≤8–16k** for multi-agent; longer context is the cost driver |
| Concurrent agents | Start at **1–2**; measure SLO before raising |
| Offload | CPU offload of large dense 70B is research-only, not the happy path |
| Isolation | **No MIG** on this GPU; use process discipline / streams / (later) Green Contexts |

Rough budget sketch:

```text
[ weights (quantized) ][ KV cache ][ engine workspace ][ OS / display / headroom ]
         |                  |              |                      |
      primary fit        context N      continuous batch      keep ≥2 GB free
```

## Inference engines observed on this host

| Engine | Path / note |
|--------|-------------|
| **Ollama** | `/usr/local/bin/ollama` · `0.30.x` · `http://127.0.0.1:11434/v1` |
| **llama-server** | Homebrew: `/home/linuxbrew/.linuxbrew/bin/llama-server` · **9190** |

See [AGENT_STACK.md](AGENT_STACK.md) for recipes and L1 flags.

## Multi-agent / CI notes

- Consumer card: no MIG partitioning.  
- Self-hosted **GPU Actions runner** for this repo is **in scope** (milestone **CI**).  
- Runner must not starve interactive desktop sessions without a clear policy (power, schedules, labels).

## Kernels (this lab is SoT)

Host GPU kernel work (L1–L3) is owned by **this repo** — see [KERNELS.md](KERNELS.md).

**Architecture:** consumer Blackwell **sm_120** is not datacenter **sm_100**. Do not assume FA4, TMEM, `tcgen05`, or **MIG**. Prefer engine Flash / FA2-class paths and L1 ablations first.

If optional neuromorphic experiments later depend on myelin PTX:

- Target **sm_120**; do **not** pin ancient PTX (e.g. 8.5) — `InvalidPtx` class failures.  
- `myelin-accelerator` is an optional upstream dep only; **not** the SoT for this host’s kernel lab.
