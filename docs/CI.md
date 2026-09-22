# CI — CPU (GitHub-hosted) + GPU (self-hosted)

| Workflow | Runner | Purpose |
|---|---|---|
| [`.github/workflows/ci-cpu.yml`](../.github/workflows/ci-cpu.yml) | **`ubuntu-latest`** | Kernel-tree checks, CUDA-disabled CMake configure, F0 correlation join (#52), clock-skew calibration (RM-1349), F0 efficiency formulas (#55), NVML capability self-test (RM-1350), and GPU-workflow policy / sanitizer-runner tests (RM-1351) |
| [`.github/workflows/ci-gpu.yml`](../.github/workflows/ci-gpu.yml) | **Self-hosted** `ShipOfTheseus` (`self-hosted`, `Linux`, `X64`, `CUDA`) | GPU probe plus sm_120 build, binaries, graph JSON, and Green Context capability/measurement JSON |
| [`.github/workflows/ci-gpu-sanitizer.yml`](../.github/workflows/ci-gpu-sanitizer.yml) | **Self-hosted** `ShipOfTheseus` (same labels) | Opt-in Compute Sanitizer (`memcheck` / `initcheck`) on first-party CUDA smokes. Manual dispatch or trusted `main` kernel-path pushes only; never pull requests |

## Host runner (this machine)

| Field | Value |
|---|---|
| Name | **ShipOfTheseus** |
| Labels | `self-hosted`, `Linux`, `X64`, `CUDA` |
| GPU | RTX 5080 (`sm_120`) |

Ensure the Actions runner is online before expecting `ci-gpu` to pick jobs:

```bash
sudo systemctl status actions.runner.*   # or: cd ~/actions-runner && ./svc.sh status
```

Re-register / label docs: [GitHub self-hosted runners](https://docs.github.com/en/actions/hosting-your-own-runners).

## Security (self-hosted)

1. **Fork PRs never run on the GPU host** — the GitHub-hosted trust gate skips the self-hosted job when `head.repo != this repo`.
2. Actions are pinned to commit SHAs, not floating tags.
3. Do not use secrets that untrusted PR code could exfiltrate on self-hosted infrastructure.
4. Desktop share: GPU jobs **serialize** with training and interactive work.
   `ci-gpu` and `ci-gpu-sanitizer` wait up to 10 minutes for ≥2048 MiB free on
   GPU 0, then fail. `ci-gpu` job timeout is **30 minutes** (wait + kernel
   smoke). `ci-gpu-sanitizer` is **45 minutes** (wait + instrumented smokes).
   Markdown-only PRs (`**/*.md`, `LICENSE`) do not schedule the GPU host.
   Compute Sanitizer never uses `pull_request`. Keep `concurrency`
   cancel-in-progress.
5. Do not store model weights or API keys in the runner work directory long-term.

## GPU headroom (serialize)

```bash
# Same wait CI uses before kernel smoke
bash kernels/tools/wait_gpu_headroom.sh 2048 600 30
```

If another compute process holds the card below 2 GiB, do not start
`agoge train-*` **or** GPU CI. Pause the self-hosted runner, or wait.
Green Context isolation is not the default for train vs Actions.

## Local equivalents

```bash
# Same as ci-cpu
cmake -S kernels -B build/kernels-cpu -DBKL_ENABLE_CUDA=OFF
python3 tools/check_f0_correlation.py \
  --markers fixtures/f0-correlation/agoge-markers.jsonl \
  --samples fixtures/f0-correlation/bkl-gpu-samples.jsonl \
  --clock-skew fixtures/f0-correlation/clock-skew.json
python3 tools/check_f0_clock_skew.py \
  --fixtures fixtures/f0-correlation/clock-skew
python3 tools/check_f0_efficiency.py
python3 tools/summarize_f0_efficiency.py \
  --markers fixtures/f0-correlation/agoge-markers.jsonl \
  --samples fixtures/f0-correlation/bkl-gpu-samples.jsonl \
  --profile-windows fixtures/f0-efficiency/profile-windows.jsonl \
  --train-fit-ref fixtures/f0-efficiency/train-fit-reference.json \
  --out results/f0-efficiency.json \
  --report results/f0-efficiency.md
python3 tools/check_nvml_capability.py
python3 tools/check_gpu_ci_policy.py
bash kernels/tools/check_compute_sanitizer_runner.sh

# CUDA smoke and first-party measurements (#17 / #19 / #21 / #30)
cmake -S kernels -B build/kernels -DBKL_ENABLE_CUDA=ON
cmake --build build/kernels -j
./build/kernels/src/bkl_device_hello
./build/kernels/src/bkl_green_ctx_bench --out results/green-ctx-bench.json
./build/kernels/src/bkl_graph_launch_bench --out results/graph-launch-bench.json
python3 -m json.tool results/graph-launch-bench.json >/dev/null

# Same report check ci-gpu runs: recomputes wave counts and per-invocation
# improvement from the report's own inputs, not just "is it valid JSON".
python3 kernels/tools/check_green_ctx_report.py results/green-ctx-bench.json

# Opt-in Compute Sanitizer (RM-1351). Same headroom wait as ci-gpu.
# --smoke is not a performance measurement. Full commands:
# recipes/compute-sanitizer.md
bash kernels/tools/run_compute_sanitizer.sh --suite \
  --bin-dir build/kernels/src \
  --out-dir results/sanitizer
```

The model-backed L1 prefix/KV experiment is intentionally manual: CI does not
pull weights or assume that a runner has the selected model resident. Run
[l1-prefix-kv-reuse.md](../recipes/l1-prefix-kv-reuse.md) on the GPU host with
an already-local model; it enforces the 2 GiB headroom rule and writes JSONL
under gitignored `results/`.

## Issue

- #11 / RM-182 — self-hosted GPU Actions runner.
- #45 / #49 — train ↔ GPU CI serialize (wait for headroom; do not dual-occupy).
- RM-1351 — opt-in Compute Sanitizer on first-party CUDA binaries.
