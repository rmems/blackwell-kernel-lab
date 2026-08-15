# blackwell-kernel-lab

**GPU kernel + local agent lab on RTX 5080 (Blackwell / sm_120)** — first-party kernel work for this host, plus on-box agents measured under a **16 GB** VRAM ceiling.

```text
This host (ShipOfTheseus) · RTX 5080 · ~16 GB GDDR7 · driver 610.x · CUDA 13.3
```

## Mission (dual track)

| Track | Goal |
|-------|------|
| **A — GPU kernels (SoT in this repo)** | L1 engine CUDA paths → L2 host scheduling → L3 first-party `.cu` / CUTLASS when L1 proves a gap. See [docs/KERNELS.md](docs/KERNELS.md). |
| **B — Local agents** | Efficient + smarter coding/tool agents: TTFT/TPOT, tool-loop, prefix/memory, multi-agent SLOs. |

This is **not** a Limen-Neural multi-repo verification lab. Training / fine-tuning stays in **agoge-forger**.

| Activity | Where it lives |
|----------|----------------|
| **GPU kernels + agent measurement (this repo)** | `rmems/blackwell-kernel-lab` |
| **Model training / fine-tune forge** | [`rmems/agoge-forger`](https://github.com/rmems/agoge-forger) |
| **Optional neuromorphic kernels (upstream)** | `Limen-Neural/myelin-accelerator` — optional dep only; **not** the SoT for this host’s kernel lab |

**Boundary contract:** [docs/FORGE_BOUNDARY.md](docs/FORGE_BOUNDARY.md) — new `.cu` here **only after L1 proves a gap**; training in the forge; no second CUDA tree under `agoge-forger/cuda/`.

## Quick start

```bash
# Host sanity
nvidia-smi
nvcc --version   # expect CUDA 13.3 on this host

# Synthetic agent-loop harness (no GPU model required)
python3 harness/agent_loop/run_synthetic.py --out results/

# Kernel/engine smoke (L1 — measures engine CUDA paths; needs Ollama up)
export BKL_BASE_URL=http://127.0.0.1:11434/v1
export BKL_MODEL=granite4.1:8b   # or another local model
python3 harness/serve/smoke_openai.py --out results/ --stream --label cell-A-baseline

# Live multi-phase agent metrics (#10): cold TTFT → tool → resume TTFT
python3 harness/agent_loop/run_live_metrics.py --out results/ --loops 1
ollama stop "$BKL_MODEL"

# L3 first-party CUDA smoke (sm_120)
cmake -S kernels -B build/kernels -DBKL_ENABLE_CUDA=ON && cmake --build build/kernels -j
./build/kernels/src/bkl_device_hello

# Docs: docs/KERNELS.md · kernels/README.md · docs/CI.md · recipes/
```

## Repo layout

```text
docs/           Mission, host baseline, kernels, stack, models, workloads, schema
configs/        Engine / agent configs (no secrets)
harness/        Scripted agent loops + reporting → results/
recipes/        Human-run playbooks
kernels/        First-party L3 CUDA workspace (sm_120) — see kernels/README.md
results/        Measurement outputs (gitignored)
```

## KPIs we track

| KPI | Why |
|-----|-----|
| Tool-loop wall time | Real agent feel (think → tool → think) |
| TTFT / TPOT p50 & p95 | Interactive streaming |
| Prefix-cache hit rate | System + tools reuse |
| VRAM peak + free headroom | 16 GB discipline |
| Concurrent agents meeting SLO | Multi-agent on one card |
| Tokens/s and energy class | Efficiency of local serve |
| Kernel / ablation deltas | L1–L3 experiments (see KERNELS.md) |

See [docs/MISSION.md](docs/MISSION.md) and [docs/RESULTS_SCHEMA.md](docs/RESULTS_SCHEMA.md).

## Milestones & version bumps

GitHub milestones include the **release tag in the title**. Closing a milestone ⇒ cut that tag + GitHub Release (notes = closed issues). Epic [#1](https://github.com/rmems/blackwell-kernel-lab/issues/1) stays open across milestones.

| Milestone | Version | Goal |
|-----------|---------|------|
| [**M0** — Lab identity + host baseline](https://github.com/rmems/blackwell-kernel-lab/milestone/1) | **v0.1.0** | AGENTS / recipes / onboarding + 16 GB host baseline |
| [**M1** — Measure stack](https://github.com/rmems/blackwell-kernel-lab/milestone/2) | **v0.2.0** | Client path, live metrics harness, inference stack, workloads, cookbook, matrix |
| [**M2** — Agent efficiency baselines](https://github.com/rmems/blackwell-kernel-lab/milestone/3) | **v0.3.0** | Telemetry, single-agent coding/tool baseline, L1 FA × graphs × quant (#16) |
| [**K0** — Kernel SoT + L3 workspace](https://github.com/rmems/blackwell-kernel-lab/milestone/4) | **v0.4.0** | Forge↔kernel boundary, `kernels/` layout, L3 smoke |
| [**M3** — Smarter multi-agent agents](https://github.com/rmems/blackwell-kernel-lab/milestone/5) | **v0.5.0** | Packing, prefix/KV, router, multi-agent SLO, constrained tools |
| [**CI** — Self-hosted GPU runner](https://github.com/rmems/blackwell-kernel-lab/milestone/6) | **patch / v0.x.0-ci** | Secure self-hosted **GPU** Actions runner (may ship mid-stream) |

**Patch** (`v0.N.M+1`): docs, recipes, extra MATRIX rows, fixups inside an open milestone — no new minor.

**CI split:** CPU workflows → GitHub-hosted `ubuntu-latest`; GPU workflows → self-hosted `ShipOfTheseus` (`CUDA` label). See [docs/CI.md](docs/CI.md).

## License

See [LICENSE](LICENSE).
