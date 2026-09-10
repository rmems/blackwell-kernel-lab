# blackwell-kernel-lab

**CUDA backend for [`rmems/agoge-forger`](https://github.com/rmems/agoge-forger)** —
local LLM fine-tuning and training on this RTX 5080 (Blackwell / sm_120, ~16 GB).

The forge owns the trainer, datasets, and SFT / QLoRA / post-training ladder.
This repo owns the GPU kernels and engine CUDA-path measurements that training
on this host actually runs on. `agoge-forger/cuda/` is a stub that points here.

```text
This host (ShipOfTheseus) · RTX 5080 · ~16 GB GDDR7 · driver 610.x · CUDA 13.3
```

## Mission

| Goal | Scope |
|------|-------|
| **Local CUDA backend for agoge-forger** | L1 engine CUDA paths → L2 host scheduling → L3 first-party `.cu` / CUTLASS when L1 proves a gap. See [docs/KERNELS.md](docs/KERNELS.md). |

This is **not** a Limen-Neural multi-repo verification lab and **not** a
multi-agent product. Training orchestration stays in the forge; the CUDA
that makes local fine-tuning honest on this 16 GB card lives here.

| Activity | Where it lives |
|----------|----------------|
| **CUDA backend / GPU kernels / engine CUDA measurements** | `rmems/blackwell-kernel-lab` |
| **Model training / fine-tune forge** | [`rmems/agoge-forger`](https://github.com/rmems/agoge-forger) |
| **Optional neuromorphic kernels (upstream)** | `Limen-Neural/myelin-accelerator` — optional dep only; **not** the SoT for this host |

**Boundary contract:** [docs/FORGE_BOUNDARY.md](docs/FORGE_BOUNDARY.md) —
new `.cu` lands here **only after L1 proves a gap**; no second CUDA tree
under `agoge-forger/cuda/`.

**Consume API for the forge:** [docs/FORGE_CONSUME.md](docs/FORGE_CONSUME.md).

## Before you train or push GPU CI

This host has **one** RTX 5080. agoge-forger training and GPU CI **serialize**.
Do not start both.

1. Run `nvidia-smi`. GPU 0 must show **≥2048 MiB free** and no unexpected
   compute process.
2. To train: do not push GPU jobs; pause the self-hosted runner if a
   `ci-gpu` run is already queued.
3. To push kernel CI: same headroom. `ci-gpu` waits up to 10 minutes for
   that floor, then fails. Markdown-only PRs do not schedule the GPU host.

Details: [docs/CI.md](docs/CI.md) · [docs/HOST_BASELINE.md](docs/HOST_BASELINE.md).

## Quick start

```bash
# Host sanity
nvidia-smi
nvcc --version   # expect CUDA 13.3 on this host

# First-party CUDA smoke and benchmarks (sm_120)
cmake -S kernels -B build/kernels -DBKL_ENABLE_CUDA=ON && cmake --build build/kernels -j
./build/kernels/src/bkl_device_hello                                        # L3 smoke
./build/kernels/src/bkl_graph_launch_bench --out results/graph-launch-bench.json  # L3
./build/kernels/src/bkl_green_ctx_bench --out results/green-ctx-bench.json        # L2

# Docs: docs/KERNELS.md · kernels/README.md · docs/CI.md · recipes/
# Forge consume: docs/FORGE_CONSUME.md
```

## Repo layout

```text
docs/           Mission, hardware baseline, kernel layering, CI, forge consume/boundary
recipes/        Human-run kernel measurement playbooks
kernels/        First-party L3 CUDA workspace (sm_120) — see kernels/README.md
results/        Kernel measurement outputs (gitignored)
```

## Kernel measurements we track

These measurements exist so agoge-forger can train and serve on this card
without guessing sm_100 / FA4 / MIG behavior.

| Measurement | Why |
|-------------|-----|
| CUDA graph launch and replay overhead | Determine when capture amortizes on sm_120 |
| Flash / fused-attention behavior | Identify prefill and KV-memory effects exposed by an engine |
| Quantized GEMM path | Measure fit and decode-bandwidth tradeoffs for 4-bit train/serve |
| Prefix / session KV reuse | Establish engine cache behavior before L2 scheduling |
| Green Context SM isolation | Measure whether L2 partitioning protects a latency-sensitive kernel |
| VRAM peak + free headroom | Keep train and kernel work within the 16 GB host limit |
| L1–L3 deltas | Justify or reject first-party kernel work |

See [docs/MISSION.md](docs/MISSION.md) and [docs/KERNELS.md](docs/KERNELS.md).
The reproducible experiments are
[recipes/l1-prefix-kv-reuse.md](recipes/l1-prefix-kv-reuse.md) (L1 prefix cache)
and [recipes/l2-green-ctx-bench.md](recipes/l2-green-ctx-bench.md) (L2 Green
Context contention).

## Milestones & version bumps

GitHub milestones include the **release tag in the title**. Closing a milestone
⇒ cut that tag + GitHub Release (notes = closed issues). Epic
[#1](https://github.com/rmems/blackwell-kernel-lab/issues/1) stays open across
milestones.

| Milestone | Version | Goal |
|-----------|---------|------|
| [**F0** — Forge CUDA backend](https://github.com/rmems/blackwell-kernel-lab/milestone/8) | **v0.2.0** | Honest local CUDA backend for agoge-forger: consume contract, 16 GB train-fit, train↔kernel coexistence |
| [**M0** — Lab identity + host baseline](https://github.com/rmems/blackwell-kernel-lab/milestone/1) | **v0.1.0** | CUDA-backend identity, onboarding, and 16 GB hardware baseline |
| [**M1** — Engine CUDA baselines](https://github.com/rmems/blackwell-kernel-lab/milestone/2) | **v0.2.0** | Reproducible L1 engine CUDA measurements and host configuration |
| [**M2** — Kernel measurement campaign](https://github.com/rmems/blackwell-kernel-lab/milestone/3) | **v0.3.0** | FlashAttention, CUDA graphs, quant-path baselines, and measured gaps |
| [**K0** — Kernel SoT + L3 workspace](https://github.com/rmems/blackwell-kernel-lab/milestone/4) | **v0.4.0** | Forge↔kernel boundary, `kernels/` layout, L3 smoke |
| [**M3** — Proven-gap kernel follow-through](https://github.com/rmems/blackwell-kernel-lab/milestone/5) | **v0.5.0** | L2 scheduling evidence and L3 experiments justified by L1 measurements |
| [**CI** — Self-hosted GPU runner](https://github.com/rmems/blackwell-kernel-lab/milestone/6) | **patch / v0.x.0-ci** | Secure self-hosted **GPU** Actions runner (may ship mid-stream) |
| [**K1** — CUDA scheduling evidence](https://github.com/rmems/blackwell-kernel-lab/milestone/7) | **v0.6.0** | L2 Green Context isolation and the L1 prefix/KV measurement (#8, #17) |

**Current cut is F0.** Older GitHub milestone titles for 2, 3, and 5 may still
carry pre-#33 agent-harness names; the names in this table are the ones that
matter. Closing F0 tags **v0.2.0** (the next real release after tagged v0.1.0).

**Patch** (`v0.N.M+1`): docs, recipes, and fixups inside an open milestone — no
new minor.

**CI split:** CPU workflows → GitHub-hosted `ubuntu-latest`; GPU workflows →
self-hosted `ShipOfTheseus` (`CUDA` label). GPU smoke waits for ≥2 GiB free,
then fails if the card stays occupied. Markdown-only PRs skip the GPU host.
See [docs/CI.md](docs/CI.md).

## License

See [LICENSE](LICENSE).
