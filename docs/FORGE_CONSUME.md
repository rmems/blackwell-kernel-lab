# Forge consume contract (contract revision 1)

**Audience:** `agoge-forger`, or anything else that wants to pin this repo as
the local CUDA backend for LLM fine-tuning and training without copying `.cu`.

**Cite:** `rmems/blackwell-kernel-lab@24039e6` (no release tag newer than
`v0.1.0` exists yet; update this pointer when one is cut). See
[`FORGE_BOUNDARY.md`](FORGE_BOUNDARY.md) for the ownership rule this contract
implements — this doc is the consume-side API, that one is the boundary.

## Host facts

From [`HOST_BASELINE.md`](HOST_BASELINE.md):

- Compute capability **12.0** (`sm_120`, Blackwell consumer) — **not** `sm_100`.
  Do not assume FA4, TMEM, `tcgen05`, or MIG.
- **16303 MiB** VRAM (~16 GB). Leave **≥2 GB free** for kernel measurement;
  see the headroom math there before sizing anything against this host.
- CUDA toolkit **13.3** at `/usr/local/cuda`.

## Layer map

From [`KERNELS.md`](KERNELS.md):

- **L1** — measure engine CUDA paths (FlashAttention, graphs, quant, prefix).
- **L2** — host scheduling (Green Context isolation, no MIG).
- **L3** — first-party `.cu` / CUTLASS, written only once L1 proves a gap.

## Runnable recipes

- [`recipes/l1-prefix-kv-reuse.md`](../recipes/l1-prefix-kv-reuse.md)
- [`recipes/l2-green-ctx-bench.md`](../recipes/l2-green-ctx-bench.md)
- [`recipes/l3-device-hello.md`](../recipes/l3-device-hello.md)
- [`recipes/l3-graph-launch-bench.md`](../recipes/l3-graph-launch-bench.md)
- [`recipes/kernel-ablation.md`](../recipes/kernel-ablation.md)

## Build and smoke

From [`kernels/README.md`](../kernels/README.md):

```bash
cmake -S kernels -B build/kernels -DBKL_ENABLE_CUDA=ON
cmake --build build/kernels -j"$(nproc)"
./build/kernels/src/bkl_device_hello
./build/kernels/src/bkl_green_ctx_bench
./build/kernels/src/bkl_graph_launch_bench
```

`-DBKL_ENABLE_CUDA=OFF` configures without CUDA (what `ci-cpu.yml` runs). GPU
build/run is on the self-hosted `ci-gpu` runner — see
[`docs/CI.md`](CI.md).

## Training ↔ GPU telemetry join (F0)

Versioned JSONL contract: [`f0-correlation-schema.md`](f0-correlation-schema.md)
(`bkl.f0_correlation.v1`). Agoge emits `agoge_marker` records (run id, phase,
step). This lab emits `bkl_gpu_sample` records. Join is `agoge_run_id` plus
monotonic time on one host. BKL does not infer train phases from GPU load.
CPU fixtures live under `fixtures/f0-correlation/`. Forge issue #144 should
emit the Agoge-owned side; the trainer is not implemented here.

## What forge must not do

- Copy `.cu` files into `agoge-forger/cuda/`. That directory stays a stub
  pointing here.
- Assume `sm_100`, FA4/TMEM/`tcgen05`, or MIG — this host is `sm_120`,
  consumer-class, single GPU.
- Grow a second first-party CUDA tree. New kernel work lands in this repo,
  not in the forge.
- Duplicate the VRAM headroom math from `HOST_BASELINE.md` into forge
  recipes; forge's own preflight (`uv run agoge check-torch`) already warns
  on the relevant threshold.
- Dual-occupy this card with GPU CI. Serialize: ≥2 GiB free, then train
  **or** `ci-gpu` (which waits up to 10 minutes then fails). See
  [`docs/CI.md`](CI.md) and the README checklist.

## QLoRA train-fit (this host)

Measured MiniCPM5 canary peak VRAM and the Granite 4.1 **unmeasured
template** live in [`TRAIN_FIT_5080.md`](TRAIN_FIT_5080.md) (#44 /
RM-1053). Forge owns the trainer YAML; this lab does not. When #53/#55
and Agoge #144 are ready, that page links the `bkl.f0_correlation.v1`
bundle instead of growing a second measurement format.

## Acceptance

A reader who has only this file should be able to find the host rules above,
run one L1 recipe and one L3 smoke binary, find the measured QLoRA train-fit
notes, and know that this repo is the CUDA backend while training
orchestration stays in `agoge-forger`.
