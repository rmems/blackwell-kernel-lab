# CI — CPU (GitHub-hosted) + GPU (self-hosted)

| Workflow | Runner | Purpose |
|---|---|---|
| [`.github/workflows/ci-cpu.yml`](../.github/workflows/ci-cpu.yml) | **`ubuntu-latest`** | Kernel-tree checks and CUDA-disabled CMake configure |
| [`.github/workflows/ci-gpu.yml`](../.github/workflows/ci-gpu.yml) | **Self-hosted** `ShipOfTheseus` (`self-hosted`, `Linux`, `X64`, `CUDA`) | GPU probe plus sm_120 L3 build, binaries, and graph-benchmark JSON |

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
4. Desktop share: GPU jobs may compete with interactive work; keep `concurrency` cancel-in-progress.
5. Do not store model weights or API keys in the runner work directory long-term.

## Local equivalents

```bash
# Same as ci-cpu
cmake -S kernels -B build/kernels-cpu -DBKL_ENABLE_CUDA=OFF

# L3 CUDA smoke (#19 / #21 / #30)
cmake -S kernels -B build/kernels -DBKL_ENABLE_CUDA=ON
cmake --build build/kernels -j
./build/kernels/src/bkl_device_hello
./build/kernels/src/bkl_graph_launch_bench --out results/graph-launch-bench.json
python3 -m json.tool results/graph-launch-bench.json >/dev/null
```

The model-backed L1 prefix/KV experiment is intentionally manual: CI does not
pull weights or assume that a runner has the selected model resident. Run
[l1-prefix-kv-reuse.md](../recipes/l1-prefix-kv-reuse.md) on the GPU host with
an already-local model; it enforces the 2 GiB headroom rule and writes JSONL
under gitignored `results/`.

## Issue

#11 / RM-182 — self-hosted GPU Actions runner for this lab.
