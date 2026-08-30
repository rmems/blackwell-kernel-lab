# Host baseline — RTX 5080 kernel lab

**Host:** ShipOfTheseus (personal workstation)  
**Refresh:** 2026-08-12T09:50Z (live `nvidia-smi` / `nvcc` on this machine)

## GPU

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce **RTX 5080** |
| VRAM | **16303 MiB** (~16 GB) |
| Compute capability | **12.0** (`sm_120`, Blackwell consumer) |
| Driver | **610.43.03** |
| Persistence | Enabled |

```bash
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,compute_cap,driver_version,persistence_mode --format=csv
nvcc --version
```

## CUDA toolkit

| Item | Value |
|---|---|
| Toolkit | **CUDA 13.3** (`/usr/local/cuda`) |
| `nvcc` | **13.3.73** (V13.3.73) |

## VRAM headroom

The card has a ~16 GB ceiling. Leave **≥2 GB free** for kernel measurement
contexts, engine workspace, and the desktop; inspect `nvidia-smi` before a GPU
run and avoid overlapping long-running workloads with CI.

Rough budget:

```text
[ engine/kernel allocation ][ CUDA context ][ OS / display ][ headroom ]
```

## Workstation / CI notes

- Consumer card: no MIG partitioning.
- The self-hosted GPU Actions runner for this repository is in scope.
- Runner jobs must not starve the desktop without a clear policy (power, schedules, labels).

## Kernels (this lab is SoT)

Host GPU kernel work (L1–L3) is owned by this repo — see [KERNELS.md](KERNELS.md).
Consumer Blackwell **sm_120** is not datacenter **sm_100**: do not assume FA4,
TMEM, `tcgen05`, or MIG. Optional myelin experiments target sm_120 and do not
pin ancient PTX.
