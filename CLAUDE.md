@AGENTS.md

# CLAUDE.md

Claude-specific notes only. Mission, ownership map, hardware limits, and PR/Linear rules are in `AGENTS.md`
(imported above); do not duplicate them here.

## Commands

Run from the repo root. "Verified" means it ran successfully in a fresh checkout on Python 3.13, CMake 3.28, shellcheck 0.11.0
and actionlint 1.7.12 (Linux, no GPU, no CUDA toolkit). CI installs no Python packages; the checks use plain `python3`.

| Command | Status |
|---|---|
| `cmake -S kernels -B build/kernels-cpu -DBKL_ENABLE_CUDA=OFF` | verified |
| `python3 tools/check_f0_correlation.py --markers fixtures/f0-correlation/agoge-markers.jsonl --samples fixtures/f0-correlation/bkl-gpu-samples.jsonl --clock-skew fixtures/f0-correlation/clock-skew.json` | verified |
| `python3 tools/check_f0_clock_skew.py --fixtures fixtures/f0-correlation/clock-skew` | verified |
| `python3 tools/check_f0_efficiency.py` | verified (21 checks) |
| `python3 tools/summarize_f0_efficiency.py --markers fixtures/f0-correlation/agoge-markers.jsonl --samples fixtures/f0-correlation/bkl-gpu-samples.jsonl --profile-windows fixtures/f0-efficiency/profile-windows.jsonl --train-fit-ref fixtures/f0-efficiency/train-fit-reference.json --print-json` | verified |
| `python3 tools/check_nvml_capability.py` | verified |
| `python3 tools/check_gpu_ci_policy.py` | verified |
| `python3 tools/test_gpu_ci_policy.py` | verified |
| `python3 tools/check_gpu_ci_gate.py` | verified (23 tests) |
| `python3 tools/test_publish_gpu_validation.py` | verified |
| `bash kernels/tools/check_compute_sanitizer_runner.sh` | verified |
| `shellcheck --format=gcc recipes/scripts/*.sh kernels/tools/*.sh` | verified |
| `./actionlint -oneline -shellcheck "$(command -v shellcheck)"` | verified (binary pinned by version and SHA-256 in `ci-lint.yml`) |
| `cmake -S kernels -B build/kernels -DBKL_ENABLE_CUDA=ON && cmake --build build/kernels -j` | not run (needs nvcc, CUDA 13.3, and the RTX 5080) |
| `./build/kernels/src/bkl_device_hello`, `bkl_graph_launch_bench --out results/graph-launch-bench.json`, `bkl_green_ctx_bench --out results/green-ctx-bench.json` | not run (need the GPU build above) |
| `python3 kernels/tools/check_green_ctx_report.py results/green-ctx-bench.json` | not run (needs the GPU benchmark's JSON output) |
| `bash kernels/tools/run_compute_sanitizer.sh --suite --bin-dir build/kernels/src --out-dir results/sanitizer` | not run (needs GPU binaries and `compute-sanitizer`) |

`ci-cpu.yml` also has a "Kernel tree is complete" shell step (every `add_executable` source and every recipe linked from
`kernels/README.md` must exist); its script was run by hand and passes. There is no repo-wide formatter or Python linter.
Before pushing, run the CPU rows that match the files you touched; `ci-gpu.yml` and `ci-gpu-sanitizer.yml` run on the self-hosted runner only.

## Architecture map

- `kernels/`: CMake workspace (`BKL_ENABLE_CUDA`, default ON) with `src/*.cu` benchmarks (`device_hello`, `graph_launch_bench`, `green_ctx_bench`). `kernels/tools/` holds the sanitizer wrapper, headroom wait, and Green Context report checker.
- `tools/`: CPU validators and report builders for the F0 correlation, clock-skew, NVML capability, and efficiency schemas, plus the GPU CI gate (`gpu_ci_gate.py`, `publish_gpu_validation.py`). `check_*.py` and `test_*.py` are the runnable entrypoints.
- `fixtures/`: JSON/JSONL inputs for those validators; schemas are described in `docs/f0-*.md`.
- `recipes/`: human-run measurement playbooks; `recipes/scripts/` holds the shell helper. `docs/CI.md` documents the three workflows.
- `results/` is gitignored except `.gitkeep`; write measurement output there.

## Invariants and non-goals

- A new `.cu` file needs an L1 gap first (`docs/FORGE_BOUNDARY.md`); do not add a second CUDA tree under `agoge-forger/cuda/`.
- Workflows and tools must never run fork or PR code on the self-hosted GPU runner. `ci-gpu.yml` uses `pull_request_target` and reads policy from the base branch; keep `persist-credentials: false` and the commit-SHA-pinned actions.
- `tools/gpu_ci_gate.py` keeps a CPU-only allowlist; new files under `tools/` count as GPU-relevant until a reviewer extends it (`docs/CI.md`).
- GPU workflows wait for at least 2 GiB free VRAM (`kernels/tools/wait_gpu_headroom.sh 2048 600 30`); keep that floor.
- NVML capability checks treat missingness as distinct from zero (`ci-cpu.yml`: "missingness != zero"); keep "missing" distinct from `0`.
- Source files here do not use SPDX headers; do not add them.
- No manifest or publish list exists (no `Cargo.toml`, `pyproject.toml`, or `package.json`), so there is nothing to exclude `CLAUDE.md` from.

## Gotchas

- Do not run the GPU rows on a shared host: training and GPU CI serialize on the one RTX 5080 (`nvidia-smi` first).
- CUDA 13.3 supports host GCC 6 to 15 only; `kernels/CMakeLists.txt` auto-discovers `g++-15` down to `g++-6` and fails otherwise.
- Never commit straight to `main`; open a PR and keep the Linear RM link in it.
- Edit only what the task needs.
