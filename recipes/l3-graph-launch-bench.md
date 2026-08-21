# Recipe: L3 CUDA graph vs eager launch (sm_120) — #30

Provides a synthetic, first-party `.cu` measurement of CUDA graph replay versus
eager launches on this host. It does **not** measure llama.cpp, a model decode,
or resolve the engine-policy question in #16.

## Build & run

```bash
cmake -S kernels -B build/kernels -DBKL_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DBKL_CUDA_HOST_COMPILER=/home/linuxbrew/.linuxbrew/bin/g++-15
cmake --build build/kernels -j"$(nproc)"
./build/kernels/src/bkl_graph_launch_bench
```

Each run writes a schema-v1 JSON artifact under `results/` (gitignored). To
choose its exact location, including in CI, pass `--out`:

```bash
./build/kernels/src/bkl_graph_launch_bench --out results/graph-launch-bench.json
python3 harness/report/aggregate.py --results results/
```

Expected banner (numbers vary):

```text
device0: NVIDIA GeForce RTX 5080 compute 12.0
bkl graph_launch_bench sm_120
empty iters=…  eager_us/launch=…  graph_us/launch=…  speedup=…
empty_chain k=32 …  eager_us/kern=…  graph_us/kern=…  speedup=…
saxpy n=1048576 iters=…  eager_us/launch=…  graph_us/launch=…  speedup=…
bkl graph_launch_bench sm_120 ok
```

Empty kernel ≈ launch overhead. Saxpy is 1M floats — still launch-ish, not a GEMM.

## Notes

- Arch: **sm_120** only. The benchmark exits with status 1 when the GPU compute capability is not 12.0.
- Not FA4 / not an agent harness.  
- The benchmark allocates two 1M-float SAXPY buffers: **8 MiB** total explicit
  device memory, plus CUDA runtime/context overhead. It is safe alongside a
  local-agent model only when the normal **>=2 GB free VRAM** headroom remains;
  check `nvidia-smi` before running it on this 16 GB host.
