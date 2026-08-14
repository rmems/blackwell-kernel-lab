# kernels/ — first-party L3 CUDA workspace (sm_120)

**Source of truth for host GPU kernels** lives in this repo (`rmems/blackwell-kernel-lab`), not in `agoge-forger/cuda/` (stub only). See [docs/FORGE_BOUNDARY.md](../docs/FORGE_BOUNDARY.md) and [docs/KERNELS.md](../docs/KERNELS.md).

## L1 vs L2 vs L3

| Layer | Meaning | Where |
|-------|---------|--------|
| **L1** | Measure engine CUDA paths (FA, graphs, quant, prefix) | `harness/`, `recipes/`, `docs/` |
| **L2** | Host scheduling (queues, isolation, later Green Context notes) | docs + harness when landed |
| **L3** | First-party `.cu` / CUTLASS **in this tree** | `kernels/` |

Do **not** land production L3 ops until L1 leaves a **proven** gap. This tree is the **on-ramp**: layout, CMake, and a minimal device smoke.

## Hard rules (this host)

- Target **`sm_120`** (RTX 5080 / compute 12.0). **sm_120 ≠ sm_100** — no FA4 / TMEM / B200 recipes.  
- CUDA toolkit on ShipOfTheseus: **13.3**.  
- No MIG.  
- Optional myelin ops stay upstream; measure here if used on this host.

## Layout

```text
kernels/
  README.md           # this file
  CMakeLists.txt      # top-level; skips cleanly without CUDA
  src/
    CMakeLists.txt
    device_hello.cu   # minimal L3 smoke (#21)
```

## Build (GPU host)

CUDA 13.3 rejects host **gcc 16+**. This tree auto-picks `g++-15` when present (Homebrew path on ShipOfTheseus), or you can set it:

```bash
cmake -S kernels -B build/kernels -DBKL_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DBKL_CUDA_HOST_COMPILER=/home/linuxbrew/.linuxbrew/bin/g++-15
cmake --build build/kernels -j"$(nproc)"
./build/kernels/src/bkl_device_hello
# expect: device0: NVIDIA GeForce RTX 5080 compute 12.0
#         bkl device_hello sm_120 ok
```

Force skip (CPU / no toolkit):

```bash
cmake -S kernels -B build/kernels -DBKL_ENABLE_CUDA=OFF
# configures successfully; no kernel targets
```

Unsupported host compiler (not recommended):

```bash
# Only if you must use system gcc 16+
cmake -S kernels -B build/kernels -DBKL_ENABLE_CUDA=ON \
  -DBKL_ALLOW_UNSUPPORTED_HOST_COMPILER=ON
```

Without g++-13/14/15 and without that opt-in, configure **fails** (no silent `-allow-unsupported-compiler`).

## CPU CI

`ci-cpu.yml` configures with `-DBKL_ENABLE_CUDA=OFF` (skip path). GPU build/run is on self-hosted `ci-gpu` (see [docs/CI.md](../docs/CI.md)).

## Recipe

[recipes/l3-device-hello.md](../recipes/l3-device-hello.md)

## Issues

- #19 / RM-487 — workspace layout  
- #21 / RM-488 — device smoke + GPU CI hook  
