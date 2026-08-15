# Recipe: L3 CUDA graph vs eager launch (sm_120) — #30

Fills the #16 gap: llama.cpp **9190** has no CUDA-graph CLI, so graphs are measured here in first-party `.cu`.

## Build & run

```bash
cmake -S kernels -B build/kernels -DBKL_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DBKL_CUDA_HOST_COMPILER=/home/linuxbrew/.linuxbrew/bin/g++-15
cmake --build build/kernels -j"$(nproc)"
./build/kernels/src/bkl_graph_launch_bench
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

- Arch: **sm_120** only.  
- Not FA4 / not an agent harness.  
