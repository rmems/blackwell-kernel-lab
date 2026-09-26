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

**Current release target: v0.2.0.** Deliver a measured first-party CUDA operator
consumed by Agoge, with repeatable training-speed gains on **both MiniCPM5 and
Granite 4.1**. Existing smoke tests, telemetry, and synthetic benchmarks are
foundations; they do not yet establish that release outcome. The delivery
sequence and acceptance gates are in [docs/RELEASE_V0_2_0.md](docs/RELEASE_V0_2_0.md).

**Consume API for the forge:** [docs/FORGE_CONSUME.md](docs/FORGE_CONSUME.md).

## Before you train or push GPU CI

This host has **one** RTX 5080. agoge-forger training and GPU CI **serialize**.
Do not start both.

1. Run `nvidia-smi`. GPU 0 must show **≥2048 MiB free** and no unexpected
   compute process.
2. To train: do not push GPU jobs; pause the self-hosted runner if a
   `ci-gpu` run is already queued.
3. To push kernel CI: same headroom. `ci-gpu` waits up to 10 minutes for
   that floor, then fails (job timeout 30 minutes including smoke).
   Markdown-only PRs do not schedule the GPU host. Compute Sanitizer is
   opt-in (`ci-gpu-sanitizer`: dispatch or trusted `main` kernel-path
   pushes) and uses the same headroom wait.

Details: [docs/CI.md](docs/CI.md) · [docs/HOST_BASELINE.md](docs/HOST_BASELINE.md) ·
[docs/TRAIN_FIT_5080.md](docs/TRAIN_FIT_5080.md) (MiniCPM5 canary peak VRAM).

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

# Docs: docs/KERNELS.md · docs/TRAIN_FIT_5080.md · kernels/README.md · docs/CI.md · recipes/
# Opt-in Compute Sanitizer: recipes/compute-sanitizer.md
# Forge consume: docs/FORGE_CONSUME.md
```

## Repo layout

```text
docs/           Mission, hardware baseline, train-fit, kernel layering, CI, forge consume/boundary
fixtures/       CPU JSON/JSONL fixtures (F0 correlation, clock skew, NVML capability, efficiency)
recipes/        Human-run kernel measurement playbooks
kernels/        First-party L3 CUDA workspace (sm_120) — see kernels/README.md
tools/          CPU validators (correlation join, clock skew, NVML capability, efficiency; no GPU in CI)
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

## Milestones & releases

New release milestones contain the version and close only when the matching
tag and release artifacts exist. All new issues originate in **Linear**;
GitHub holds implementation PRs and existing issue mirrors.

**Tagged so far:** [`v0.1.0`](https://github.com/rmems/blackwell-kernel-lab/releases/tag/v0.1.0)
(M0 lab identity + host baseline). Epic
[#1](https://github.com/rmems/blackwell-kernel-lab/issues/1) stays open across
tracks.

| Milestone | Role |
|-----------|------|
| [**v0.2.0 — First Agoge CUDA operator: MiniCPM5 + Granite**](https://github.com/rmems/blackwell-kernel-lab/milestone/8) | Active release: profile both workloads, freeze a justified operator, implement/package, integrate, qualify, then tag |
| [**M0** — Lab identity + host baseline](https://github.com/rmems/blackwell-kernel-lab/milestone/1) | Closed; **tagged v0.1.0** |
| [**M1** — Engine CUDA / measure stack](https://github.com/rmems/blackwell-kernel-lab/milestone/2) | Historical, closed; L1 engine measurements and host configuration |
| [**M2** — Kernel measurement campaign](https://github.com/rmems/blackwell-kernel-lab/milestone/3) | Historical, closed; FA, graphs, quant-path baselines |
| [**K0** — Kernel SoT + L3 workspace](https://github.com/rmems/blackwell-kernel-lab/milestone/4) | Historical, closed; boundary, workspace, and smoke |
| [**M3** — Retired](https://github.com/rmems/blackwell-kernel-lab/milestone/5) | Closed not-planned (agent-product hallucination); never a release |
| [**CI** — Self-hosted GPU runner](https://github.com/rmems/blackwell-kernel-lab/milestone/6) | Historical, closed; GPU runner foundations |
| [**K1** — CUDA scheduling evidence](https://github.com/rmems/blackwell-kernel-lab/milestone/7) | Historical, closed; L2 scheduling evidence |

M1/M2/K0/CI/K1 were retired as historical work tracks on 2026-09-22, an explicit
one-time exception that creates no retrospective releases. Their issue history
is preserved. v0.2.0 stays open until both model gates and release delivery pass.

**CI split:** CPU workflows → GitHub-hosted `ubuntu-latest`; GPU workflows →
self-hosted `ShipOfTheseus` (`CUDA` label). The hosted **GPU validation** check
always reports; known documentation/CPU-only changes skip the GPU host, while
kernel/build/binding/GPU-policy changes require successful GPU smoke. Smoke
waits for ≥2 GiB free, then fails if the card stays occupied.
See [docs/CI.md](docs/CI.md).

## License

See [LICENSE](LICENSE).
