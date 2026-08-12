# Host baseline — local agents on RTX 5080

**Host:** ShipOfTheseus (personal workstation)  
**Refresh:** 2026-08-11  

## GPU

| Item | Value |
|------|--------|
| GPU | NVIDIA GeForce **RTX 5080** |
| VRAM | **16303 MiB** (~16 GB) |
| Compute capability | **12.0** (`sm_120`, Blackwell consumer) |
| Driver | **610.43.03** (verify with `nvidia-smi`) |
| Persistence | Prefer Persistence-M **On** for long agent runs |

```bash
nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv
```

## CUDA toolkit

| Item | Value |
|------|--------|
| Toolkit | **CUDA 13.3** (`/usr/local/cuda`) |
| `nvcc` | 13.3.x |

```bash
nvcc --version
```

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
| **Ollama** | `/usr/local/bin/ollama` |
| **llama-server** | Homebrew: `/home/linuxbrew/.linuxbrew/bin/llama-server` |

See [AGENT_STACK.md](AGENT_STACK.md) for recipes.

## Multi-agent / CI notes

- Consumer card: no MIG partitioning.  
- Self-hosted **GPU Actions runner** for this repo is **in scope** (see epic CI milestone).  
- Runner must not starve interactive desktop sessions without a clear policy (power, schedules, labels).

## Kernels (this lab is SoT)

Host GPU kernel work (L1–L3) is owned by **this repo** — see [KERNELS.md](KERNELS.md).

**Architecture:** consumer Blackwell **sm_120** is not datacenter **sm_100**. Do not assume FA4, TMEM, `tcgen05`, or **MIG**. Prefer engine Flash / FA2-class paths and L1 ablations first.

If optional neuromorphic experiments later depend on myelin PTX:

- Target **sm_120**; do **not** pin ancient PTX (e.g. 8.5) — `InvalidPtx` class failures.  
- `myelin-accelerator` is an optional upstream dep only; **not** the SoT for this host’s kernel lab.
