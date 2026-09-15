# Recipe: Compute Sanitizer (sm_120) — RM-1351

Bounded NVIDIA Compute Sanitizer checks for the first-party CUDA binaries in
this repo. This is a **memory / init correctness** gate, not a performance
profile and not a proof of algorithmic correctness.

Sanitizer work shares the RTX 5080 with training and `ci-gpu`. Leave **≥2 GiB
free** on this 16 GB host before starting; Compute Sanitizer adds shadow-memory
overhead on top of the binary's own allocations.

## When this runs

- **Manual:** `workflow_dispatch` on
  [`.github/workflows/ci-gpu-sanitizer.yml`](../.github/workflows/ci-gpu-sanitizer.yml)
  (write access to this repo).
- **Trusted branch:** push to `main` that touches `kernels/src/**` or the
  sanitizer workflow/runner.
- **Never:** pull requests (fork or same-repo), markdown-only cuts, or
  GitHub-hosted CPU jobs. Fork code does not run on the self-hosted GPU.

## Build

Same flag-free CMake as `ci-gpu` / [l3-device-hello.md](l3-device-hello.md):

```bash
bash kernels/tools/wait_gpu_headroom.sh 2048 600 30
cmake -S kernels -B build/kernels -DBKL_ENABLE_CUDA=ON
cmake --build build/kernels -j"$(nproc)"
```

## Reproduce each sanitizer command

The suite runner is what CI invokes. It records CUDA / driver / GPU, the
binary SHA-256, the exact command, and `compute-sanitizer --version` next to a
byte-bounded log.

```bash
bash kernels/tools/run_compute_sanitizer.sh --suite \
  --bin-dir build/kernels/src \
  --out-dir results/sanitizer
```

Equivalent one-liners (each must exit 0 with `ERROR SUMMARY: 0 errors`):

```bash
# memcheck: out-of-bounds / misaligned device access (all three binaries)
compute-sanitizer --tool memcheck --error-exitcode 1 --print-limit 50 \
  --check-exit-code yes \
  ./build/kernels/src/bkl_device_hello

compute-sanitizer --tool memcheck --error-exitcode 1 --print-limit 50 \
  --check-exit-code yes \
  ./build/kernels/src/bkl_graph_launch_bench \
  --smoke --out results/sanitizer/graph-launch-memcheck.json

compute-sanitizer --tool memcheck --error-exitcode 1 --print-limit 50 \
  --check-exit-code yes \
  ./build/kernels/src/bkl_green_ctx_bench \
  --smoke --out results/sanitizer/green-ctx-memcheck.json

# initcheck: uninitialized global reads (hello + SAXPY path)
compute-sanitizer --tool initcheck --error-exitcode 1 --print-limit 50 \
  --check-exit-code yes \
  ./build/kernels/src/bkl_device_hello

compute-sanitizer --tool initcheck --error-exitcode 1 --print-limit 50 \
  --check-exit-code yes \
  ./build/kernels/src/bkl_graph_launch_bench \
  --smoke --out results/sanitizer/graph-launch-initcheck.json
```

`--error-exitcode 1` is required: Compute Sanitizer defaults to 0 even when it
reports errors, which would hide findings in CI.

`--smoke` is **not** a performance measurement. It keeps graph-launch and Green
Context launches bounded under instrumentation. Do not feed smoke JSON to
`kernels/tools/check_green_ctx_report.py` (that checker requires the full
3×7 sample report).

## Tools we do not run in CI

| Tool | Why it is not in the suite |
|---|---|
| `racecheck` | Reports shared-memory races. None of the first-party kernels use `__shared__`. Cost is typically far above memcheck. |
| `synccheck` | Reports invalid `__syncthreads` / barrier use. These binaries do not use those primitives. |

Re-evaluate if a kernel grows shared memory or cooperative groups.

## Known-good smoke evidence

CPU CI does not have a GPU. The pass/fail contract is checked on
GitHub-hosted runners with fake tools:

```bash
python3 tools/check_gpu_ci_policy.py
bash kernels/tools/check_compute_sanitizer_runner.sh
```

Log format fixtures (not a substitute for a ShipOfTheseus run):

- [`fixtures/compute-sanitizer/memcheck-clean.log`](../fixtures/compute-sanitizer/memcheck-clean.log)
- [`fixtures/compute-sanitizer/memcheck-finding.log`](../fixtures/compute-sanitizer/memcheck-finding.log)

On the GPU host, a clean run prints `ERROR SUMMARY: 0 errors` and writes
`results/sanitizer/summary.json`. Dispatch `ci-gpu-sanitizer` on ShipOfTheseus
for live evidence; logs upload as the `compute-sanitizer-logs` artifact (14-day
retention, 1 MiB cap per log).

## Intentionally faulty example (docs only)

Do **not** add this kernel to `kernels/src` or CMake. It exists so the finding
fixture and this recipe have a concrete memcheck failure mode:

```cuda
// Docs-only. Identifier bkl_oob_example must not appear under kernels/src.
__global__ void bkl_oob_example(int* p) {
  p[threadIdx.x + (1 << 20)] = 1;  // out-of-bounds write
}
```

`compute-sanitizer --tool memcheck --error-exitcode 1` would report
`ERROR SUMMARY: 1 error` and fail CI. CPU tests refuse to ship that identifier
as a production target.

## Non-goals

- Untrusted PR execution on the self-hosted GPU
- Automatic rewriting of kernel source
- Full Nsight / nsys profiling
- Claiming sanitizer proves the algorithm is correct
