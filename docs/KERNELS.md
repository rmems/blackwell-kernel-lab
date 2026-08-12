# Kernels for local agents on RTX 5080

This lab works on **inference/runtime kernels** that make **local agents** faster and more stable on **sm_120 / 16 GB** — not neuromorphic CUDA, and not greenfield attention from day one.

## Layers

| Layer | Meaning | First work |
|-------|---------|------------|
| **L1** | Use & measure engine CUDA paths (FlashAttn, CUDA graphs, Q4/NVFP4 GEMM, prefix cache) | **Yes — start here** |
| **L2** | Host scheduling (Green Contexts, dual P/D queues) | After L1 baselines |
| **L3** | New `.cu` / CUTLASS kernels | Only if L1 leaves a proven gap |

## Why not “write CUDA first”?

The high-impact kernels already run **inside** Ollama / llama.cpp (and peers). Writing new device code before measuring FA × graphs × quant is optimizing blind.

**The kernels are CUDA.** First lab code is usually **Python/CLI driving those kernels** and recording agent KPIs.

## Ranked L1 levers (agent loops)

1. **4-bit weight path** (Q4 / MXFP4 / NVFP4 when native)  
2. **CUDA graphs** (short tool decode)  
3. **Flash / fused attention** (engine FA — **not** datacenter FA4 / sm_100)  
4. **Prefix / session KV reuse** (system+tools first)  
5. Later: Green Contexts for multi-agent TPOT  

Hard rule: **sm_120 ≠ sm_100**. No FA4/TMEM assumptions. No MIG.

## First campaign

| Issue | Role |
|-------|------|
| #12 / RM-476 | Engine up — unlocks GPU kernels |
| #16 / RM-470 | **Main kernel experiment:** FA × graphs × quant ablations |
| #10 / RM-184 | Record TTFT/TPOT/VRAM for each cell |

See [recipes/kernel-ablation.md](../recipes/kernel-ablation.md) and [recipes/engine-smoke.md](../recipes/engine-smoke.md).
