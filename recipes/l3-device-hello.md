# Recipe: L3 device hello (sm_120) — #19 / #21

Proves first-party CUDA in this repo builds and runs on ShipOfTheseus.

## Build & run

```bash
# CUDA 13.3 + host gcc: prefer g++-15 (system gcc 16 is too new)
cmake -S kernels -B build/kernels -DBKL_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DBKL_CUDA_HOST_COMPILER=/home/linuxbrew/.linuxbrew/bin/g++-15
cmake --build build/kernels -j"$(nproc)"
./build/kernels/src/bkl_device_hello
```

Expected (RTX 5080):

```text
device0: NVIDIA GeForce RTX 5080 compute 12.0
bkl device_hello sm_120 ok
```

## CPU-only skip

```bash
cmake -S kernels -B build/kernels-cpu -DBKL_ENABLE_CUDA=OFF
# no bkl_device_hello target — configure succeeds
```

## Notes

- Arch: **sm_120** only in this lab’s CMake.  
- Not FA4 / not myelin copy-paste.  
- Production ops need an L1-proven gap first (`docs/KERNELS.md`).  
