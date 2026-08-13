# CI — CPU (GitHub-hosted) + GPU (self-hosted)

| Workflow | Runner | Purpose |
|----------|--------|---------|
| [`.github/workflows/ci-cpu.yml`](../.github/workflows/ci-cpu.yml) | **`ubuntu-latest`** (GitHub servers) | `py_compile` + synthetic harness — no GPU |
| [`.github/workflows/ci-gpu.yml`](../.github/workflows/ci-gpu.yml) | **Self-hosted** `ShipOfTheseus` labels: `self-hosted`, `Linux`, `X64`, `CUDA` | `nvidia-smi` + synthetic; optional live smoke |

## Host runner (this machine)

| Field | Value |
|-------|--------|
| Name | **ShipOfTheseus** |
| Labels | `self-hosted`, `Linux`, `X64`, `CUDA` |
| GPU | RTX 5080 (`sm_120`) |

Ensure the Actions runner service is **online** before expecting `ci-gpu` to pick jobs:

```bash
# Typical user install location (adjust if different)
sudo systemctl status actions.runner.*   # or: cd ~/actions-runner && ./svc.sh status
```

Re-register / label docs: [GitHub self-hosted runners](https://docs.github.com/en/actions/hosting-your-own-runners).

## Security (self-hosted)

1. **Fork PRs never run on the GPU host** — trust gate job runs on **GitHub-hosted** `ubuntu-latest` and sets `allow=false` when `head.repo != this repo`. The self-hosted job is skipped entirely (no checkout of fork code on the GPU box).  
2. Actions are **pinned to commit SHAs** (not floating tags) in workflow files.  
3. Prefer **not** using secrets that can be exfiltrated by untrusted PR code on self-hosted.  
4. Optional live smoke (`vars.BKL_GPU_LIVE_SMOKE=1`) only when Ollama is intentionally left up.  
5. Desktop share: GPU jobs may compete with interactive agents — keep `concurrency` cancel-in-progress.  
6. Do not store model weights or API keys in the runner work dir long-term.

## Local equivalents

```bash
# Same as ci-cpu
python3 -m py_compile harness/agent_loop/*.py harness/serve/*.py harness/report/*.py
python3 harness/agent_loop/run_synthetic.py --out results/

# Live metrics (#10) — needs Ollama/llama-server
export BKL_BASE_URL=http://127.0.0.1:11434/v1
export BKL_MODEL=granite4.1:8b
python3 harness/agent_loop/run_live_metrics.py --out results/ --loops 1
```

## Issue

#11 / RM-182 — self-hosted GPU Actions runner for this lab.
