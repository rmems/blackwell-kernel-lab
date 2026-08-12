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
ollama stop "$BKL_MODEL"

# Docs: docs/KERNELS.md · docs/AGENT_STACK.md · recipes/engine-smoke.md
```

## Repo layout

```text
docs/           Mission, host baseline, kernels, stack, models, workloads, schema
configs/        Engine / agent configs (no secrets)
harness/        Scripted agent loops + reporting → results/
recipes/        Human-run playbooks
kernels/        First-party L3 CUDA / CUTLASS (when landed; see KERNELS.md)
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

## Milestones

| | Goal |
|--|------|
| **M0** | Identity + host baseline for local agents |
| **M1** | Harness + matrix + inference stack + model cookbook |
| **M2** | Agents run on-box with measured efficiency baselines |
| **M3** | Smarter multi-turn / multi-agent / constrained tools |
| **K0** | Kernel SoT declared; L1 ablations (#16); L3 workspace ready |
| **CI** | Self-hosted **GPU** Actions runner for this lab |

## License

See [LICENSE](LICENSE).
