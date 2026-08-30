# Kernels for RTX 5080 (SoT in this repo)

**`rmems/blackwell-kernel-lab` is the source of truth for GPU kernel work on
this host** (sm_120 / 16 GB): engine CUDA paths (L1), host scheduling (L2), and
first-party `.cu` / CUTLASS (L3) when L1 proves a gap.

Keep at least **2 GiB of VRAM free** before a measurement on this 16 GB host;
see [HOST_BASELINE.md](HOST_BASELINE.md) for the operational headroom rule.

Not neuromorphic-first. Not greenfield attention from day one. Training remains
in `agoge-forger` (its `cuda/` tree is a stub only).

**Boundary contract:** [FORGE_BOUNDARY.md](FORGE_BOUNDARY.md).

## Layers

| Layer | Meaning | First work |
|---|---|---|
| **L1** | Use and measure engine CUDA paths (FlashAttn, CUDA graphs, Q4/NVFP4 GEMM, prefix cache) | **Yes — start here** |
| **L2** | Host scheduling (Green Contexts, dual P/D queues) | After L1 baselines |
| **L3** | New `.cu` / CUTLASS kernels in this repo | Only if L1 leaves a proven gap |

## Why not “write CUDA first”?

The high-impact kernels already run inside Ollama / llama.cpp (and peers).
Writing device code before measuring FA × graphs × quant is optimizing blind.
L1 measures engine CUDA primitives and their observable kernel behavior; L3 code
lands under `kernels/`, never under `agoge-forger/cuda/`.

## Ranked L1 levers

1. **4-bit weight path** (Q4 / MXFP4 / NVFP4 when native)
2. **CUDA graphs** (launch and replay behavior)
3. **Flash / fused attention** (engine FA — not datacenter FA4 / sm_100)
4. **Prefix / session KV reuse** (engine cache behavior)
5. Later: Green Contexts for host scheduling

Hard rule: **sm_120 ≠ sm_100**. No FA4/TMEM assumptions. No MIG.

## Campaign status

| Issue | Role | Status |
|---|---|---|
| #12 / RM-476 | Engine up — unlocks GPU kernels | Done |
| #16 / RM-470 | L1 FA × graphs × quant ablations | Done (graphs = engine-default on llama.cpp 9190; D deferred) |
| #8 / RM-183 | L1 prefix / KV reuse | **Measured** (prefix-first: 21.70× prompt-eval median) |
| #20 / RM-486 | Forge ↔ kernel boundary contract | Done |
| #11 / RM-182 | Self-hosted GPU Actions runner | Done (#27) |
| #19 / RM-487 | L3 `kernels/` workspace layout | **This tree** (`kernels/`) |
| #21 / RM-488 | L3 device-hello smoke + GPU CI | **This tree** (`bkl_device_hello`) |
| #30 | L3 CUDA graph vs eager launch (sm_120) | **This tree** (`bkl_graph_launch_bench`) |

Measured on ShipOfTheseus (RTX 5080, CUDA 13.3, 2026-08-20): synthetic empty
and 1M SAXPY launches are slower in a graph (0.62× and 0.89×), while a captured
32-kernel chain replays at 2.87× versus eager. These are not model/decode
measurements and do not themselves choose an inference-engine graph policy.

Measured on the same host with Ollama 0.33.2 and a 100%-GPU-resident
`phi4:14b` (2026-08-30): placing the stable policy before the varying task in a
1,466-token prompt reduced median prompt evaluation from 348.162 ms to 16.041
ms (95.39%, 21.70×) across three independent invocations. Streaming resume
TTFT fell from 407.302 ms to 19.452 ms. All three cleared the 10% gate while
retaining 3,481–3,509 MiB of free VRAM. Policy: keep byte-stable
system/tool material first, drop stale request-specific material, and reuse
only when model, engine, context, options, and policy match. See
[l1-prefix-kv-reuse.md](../recipes/l1-prefix-kv-reuse.md).

L3 workspace: [kernels/README.md](../kernels/README.md). Recipes:
[L1 prefix/KV reuse](../recipes/l1-prefix-kv-reuse.md) ·
[L3 device hello](../recipes/l3-device-hello.md) ·
[L3 graph-launch benchmark](../recipes/l3-graph-launch-bench.md).
