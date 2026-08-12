# Kernels for RTX 5080 (SoT in this repo)

**`rmems/blackwell-kernel-lab` is the source of truth for GPU kernel work on this host** (sm_120 / 16 GB). That includes measuring engine CUDA paths (L1), host scheduling (L2), and first-party `.cu` / CUTLASS (L3) when L1 proves a gap.

Not neuromorphic-first. Not greenfield attention from day one. Training forge stays in `agoge-forger` (its `cuda/` tree is a stub only).

**Boundary contract (agents must follow):** [FORGE_BOUNDARY.md](FORGE_BOUNDARY.md) — where `.cu` lands, what the forge may consume, decision tree.

## Layers

| Layer | Meaning | First work |
|-------|---------|------------|
| **L1** | Use & measure engine CUDA paths (FlashAttn, CUDA graphs, Q4/NVFP4 GEMM, prefix cache) | **Yes — start here** |
| **L2** | Host scheduling (Green Contexts, dual P/D queues) | After L1 baselines |
| **L3** | New `.cu` / CUTLASS kernels **in this repo** | Only if L1 leaves a proven gap |

## Why not “write CUDA first”?

The high-impact kernels already run **inside** Ollama / llama.cpp (and peers). Writing new device code before measuring FA × graphs × quant is optimizing blind.

**The kernels are CUDA.** First lab code is usually **Python/CLI driving those kernels** and recording agent KPIs. L3 code lands under `kernels/` (or equivalent) here — not under `agoge-forger/cuda/`.

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
