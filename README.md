# blackwell-kernel-lab

**GPU kernel lab for the RTX 5080 (Blackwell / sm_120)** — the source of truth
for first-party kernel work and engine CUDA-path measurements on this host.

```text
This host (ShipOfTheseus) · RTX 5080 · ~16 GB GDDR7 · driver 610.x · CUDA 13.3
```

## Mission

| Goal | Scope |
|------|-------|
| **GPU kernels (SoT in this repo)** | L1 engine CUDA paths → L2 host scheduling → L3 first-party `.cu` / CUTLASS when L1 proves a gap. See [docs/KERNELS.md](docs/KERNELS.md). |

This is **not** a Limen-Neural multi-repo verification lab. Training / fine-tuning
stays in **agoge-forger**.

| Activity | Where it lives |
|----------|----------------|
| **GPU kernel work and engine CUDA measurements** | `rmems/blackwell-kernel-lab` |
| **Model training / fine-tune forge** | [`rmems/agoge-forger`](https://github.com/rmems/agoge-forger) |
| **Optional neuromorphic kernels (upstream)** | `Limen-Neural/myelin-accelerator` — optional dep only; **not** the SoT for this host’s kernel lab |

**Boundary contract:** [docs/FORGE_BOUNDARY.md](docs/FORGE_BOUNDARY.md) — new
`.cu` lands here **only after L1 proves a gap**; training remains in the forge;
no second CUDA tree under `agoge-forger/cuda/`.

## Quick start

```bash
# Host sanity
nvidia-smi
nvcc --version   # expect CUDA 13.3 on this host

# L3 first-party CUDA smoke (sm_120)
cmake -S kernels -B build/kernels -DBKL_ENABLE_CUDA=ON && cmake --build build/kernels -j
./build/kernels/src/bkl_device_hello
./build/kernels/src/bkl_graph_launch_bench --out results/graph-launch-bench.json

# Docs: docs/KERNELS.md · kernels/README.md · docs/CI.md · recipes/
```

## Repo layout

```text
docs/           Mission, hardware baseline, kernel layering, CI, boundary
recipes/        Human-run kernel measurement playbooks
kernels/        First-party L3 CUDA workspace (sm_120) — see kernels/README.md
results/        Kernel measurement outputs (gitignored)
```

## Kernel measurements we track

| Measurement | Why |
|-------------|-----|
| CUDA graph launch and replay overhead | Determine when capture amortizes on sm_120 |
| Flash / fused-attention behavior | Identify prefill and KV-memory effects exposed by an engine |
| Quantized GEMM path | Measure fit and decode-bandwidth tradeoffs |
| Prefix / session KV reuse | Establish engine cache behavior before L2 scheduling |
| VRAM peak + free headroom | Keep experiments within the 16 GB host limit |
| L1–L3 deltas | Justify or reject first-party kernel work |

See [docs/MISSION.md](docs/MISSION.md) and [docs/KERNELS.md](docs/KERNELS.md).
The reproducible L1 prefix-cache experiment is
[recipes/l1-prefix-kv-reuse.md](recipes/l1-prefix-kv-reuse.md).

## Milestones & version bumps

GitHub milestones include the **release tag in the title**. Closing a milestone
⇒ cut that tag + GitHub Release (notes = closed issues). Epic
[#1](https://github.com/rmems/blackwell-kernel-lab/issues/1) stays open across
milestones.

| Milestone | Version | Goal |
|-----------|---------|------|
| [**M0** — Lab identity + host baseline](https://github.com/rmems/blackwell-kernel-lab/milestone/1) | **v0.1.0** | Kernel-lab identity, onboarding, and 16 GB hardware baseline |
| [**M1** — Engine CUDA baselines](https://github.com/rmems/blackwell-kernel-lab/milestone/2) | **v0.2.0** | Reproducible L1 engine CUDA measurements and host configuration |
| [**M2** — Kernel measurement campaign](https://github.com/rmems/blackwell-kernel-lab/milestone/3) | **v0.3.0** | FlashAttention, CUDA graphs, quant-path baselines, and measured gaps |
| [**K0** — Kernel SoT + L3 workspace](https://github.com/rmems/blackwell-kernel-lab/milestone/4) | **v0.4.0** | Forge↔kernel boundary, `kernels/` layout, L3 smoke |
| [**M3** — Proven-gap kernel follow-through](https://github.com/rmems/blackwell-kernel-lab/milestone/5) | **v0.5.0** | L2 scheduling evidence and L3 experiments justified by L1 measurements |
| [**CI** — Self-hosted GPU runner](https://github.com/rmems/blackwell-kernel-lab/milestone/6) | **patch / v0.x.0-ci** | Secure self-hosted **GPU** Actions runner (may ship mid-stream) |

**Patch** (`v0.N.M+1`): docs, recipes, and fixups inside an open milestone — no
new minor.

**CI split:** CPU workflows → GitHub-hosted `ubuntu-latest`; GPU workflows →
self-hosted `ShipOfTheseus` (`CUDA` label). See [docs/CI.md](docs/CI.md).

## License

See [LICENSE](LICENSE).
