# blackwell-kernel-lab

**Local agent lab on RTX 5080 (Blackwell / sm_120)** — run agents on-box, measure agent-loop KPIs, and experiment with inference efficiency under a **16 GB** VRAM ceiling.

```text
This host (ShipOfTheseus) · RTX 5080 · ~16 GB GDDR7 · driver 610.x · CUDA 13.3
```

## Mission

Make **local AI agents** on this machine:

1. **More efficient** — TTFT, TPOT, tool-loop latency, VRAM headroom, energy, multi-agent stability  
2. **Smarter** — reliable tool calling, prefix/memory reuse, constrained JSON, concurrent-agent SLOs  

This is **not** a Limen-Neural multi-repo verification lab. Upstream neuromorphic crates may appear later as *optional* control-plane experiments only.

| Activity | Where it lives |
|----------|----------------|
| **Local agents + measurement (this repo)** | `rmems/blackwell-kernel-lab` |
| **Model training / fine-tune forge** | [`rmems/agoge-forger`](https://github.com/rmems/agoge-forger) |
| **Blackwell CUDA kernels (neuromorphic)** | `Limen-Neural/myelin-accelerator` (optional dep) |

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
docs/           Mission, host baseline, stack, models, workloads, schema
configs/        Engine / agent configs (no secrets)
harness/        Scripted agent loops + reporting → results/
recipes/        Human-run playbooks
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

See [docs/MISSION.md](docs/MISSION.md) and [docs/RESULTS_SCHEMA.md](docs/RESULTS_SCHEMA.md).

## Milestones

| | Goal |
|--|------|
| **M0** | Identity + host baseline for local agents |
| **M1** | Harness + matrix + inference stack + model cookbook |
| **M2** | Agents run on-box with measured efficiency baselines |
| **M3** | Smarter multi-turn / multi-agent / constrained tools |
| **CI** | Self-hosted **GPU** Actions runner for this lab |

## License

See [LICENSE](LICENSE).
