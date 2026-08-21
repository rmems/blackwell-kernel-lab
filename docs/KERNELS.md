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

## Campaign status

| Issue | Role | Status |
|-------|------|--------|
| #12 / RM-476 | Engine up — unlocks GPU kernels | Done |
| #16 / RM-470 | L1 FA × graphs × quant ablations | Done (graphs = engine-default on llama.cpp 9190; D deferred) |
| #20 / RM-486 | Forge ↔ kernel boundary contract | Done |
| #10 / RM-184 | Live agent metrics harness | Done (#27) |
| #11 / RM-182 | Self-hosted GPU Actions runner | Done (#27) |
| #19 / RM-487 | L3 `kernels/` workspace layout | **This tree** (`kernels/`) |
| #21 / RM-488 | L3 device-hello smoke + GPU CI | **This tree** (`bkl_device_hello`) |
| #30 | L3 CUDA graph vs eager launch (sm_120) | **This tree** (`bkl_graph_launch_bench`) — synthetic launch-overhead measurement |

Measured on ShipOfTheseus (RTX 5080, CUDA 13.3, 2026-08-15): in these
**synthetic** workloads, wrapping one launch in a graph is slower (empty 0.60×,
1M SAXPY 0.89×), while capturing a **32-kernel chain** and replaying it is
2.78× versus that eager chain. These are not model/decode measurements, so
they do not by themselves select an inference-engine graph policy or resolve
#16.

L3 workspace: [kernels/README.md](../kernels/README.md) · recipe [l3-device-hello.md](../recipes/l3-device-hello.md) · [l3-graph-launch-bench.md](../recipes/l3-graph-launch-bench.md).

See [recipes/kernel-ablation.md](../recipes/kernel-ablation.md), [recipes/engine-smoke.md](../recipes/engine-smoke.md), [FORGE_BOUNDARY.md](FORGE_BOUNDARY.md).
