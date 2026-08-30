# Mission — RTX 5080 GPU kernel lab

## Goal

Own GPU kernel work for this RTX 5080 host: L1 engine CUDA paths, L2 scheduling,
and L3 first-party kernels only when L1 proves a gap ([KERNELS.md](KERNELS.md)).

## Non-goals

| Non-goal | Owner instead |
|---|---|
| Multi-repo Limen CUDA build matrix | Retired from this epic |
| Model training / fine-tuning forge | [`rmems/agoge-forger`](https://github.com/rmems/agoge-forger) |
| Blind L3 kernel writing without L1 baselines | Deferred until a measured gap |

## Ownership (kernels)

| Layer | Meaning | SoT |
|---|---|---|
| L1 | Measure engine CUDA (FA, graphs, quant, prefix) | **this repo** |
| L2 | Host scheduling (Green Contexts, queues) | **this repo** |
| L3 | New `.cu` / CUTLASS | **this repo** (when justified) |
| Neuromorphic optional | myelin-style ops | `Limen-Neural/myelin-accelerator` as optional dep only |

`agoge-forger/cuda/` remains a stub; real kernel lab work lands here.

## Success

- [x] README/epic describe a GPU kernel source of truth, not Limen verification.
- [x] Hardware baseline documented for the 16 GB RTX 5080.
- [x] L1 engine-CUDA measurement methodology documented ([kernel ablation recipe](../recipes/kernel-ablation.md)).
- [x] L3 workspace / smoke path (`kernels/`, `bkl_device_hello`, sm_120).
- [x] Self-hosted GPU CI runner for this lab (#11 / #27).

## Research anchors

- Quant fit → CUDA graphs → engine Flash (not FA4/B200) → prefix reuse → optional Green Contexts.
- Proven gap → L3 first-party kernel.
- CUDA graph launch/replay measurements provide engine-measurement context, not an inference-policy claim.
