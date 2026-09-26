# Mission — CUDA backend for agoge-forger (RTX 5080)

## Goal

Be the **local CUDA backend** so [`rmems/agoge-forger`](https://github.com/rmems/agoge-forger)
can fine-tune and train LLMs honestly on this RTX 5080: L1 engine CUDA paths,
L2 scheduling, and L3 first-party kernels only when L1 proves a gap
([KERNELS.md](KERNELS.md), [FORGE_CONSUME.md](FORGE_CONSUME.md)).

## Non-goals

| Non-goal | Owner instead |
|---|---|
| Multi-repo Limen CUDA build matrix | Retired from this epic |
| Trainer, datasets, SFT / QLoRA ladder | [`rmems/agoge-forger`](https://github.com/rmems/agoge-forger) |
| Blind L3 kernel writing without L1 baselines | Deferred until a measured gap |
| Multi-agent / local-agent product | Retired (M3 not-planned) |

## Ownership (kernels)

| Layer | Meaning | SoT |
|---|---|---|
| L1 | Measure engine CUDA (FA, graphs, quant, prefix) | **this repo** |
| L2 | Host scheduling (Green Contexts, queues) | **this repo** |
| L3 | New `.cu` / CUTLASS | **this repo** (when justified) |
| Neuromorphic optional | myelin-style ops | `Limen-Neural/myelin-accelerator` as optional dep only |

`agoge-forger/cuda/` remains a stub; the CUDA backend for local training
lands here.

## Success

**Upcoming release target:** [v0.2.0](RELEASE_V0_2_0.md) must deliver a BKL-owned
CUDA operator used by Agoge with a measured training-speed improvement on
both MiniCPM5 and Granite 4.1. The following completed foundations do not
by themselves establish a reusable training operator or its performance.

- [x] README/epic describe the agoge-forger CUDA backend, not a Limen or agent playground.
- [x] Hardware baseline documented for the 16 GB RTX 5080.
- [x] L1 engine-CUDA measurement methodology documented ([kernel ablation recipe](../recipes/kernel-ablation.md)).
- [x] L3 workspace / smoke path (`kernels/`, `bkl_device_hello`, sm_120).
- [x] Self-hosted GPU CI runner for this lab (#11 / #27).

## Research anchors

- Quant fit → CUDA graphs → engine Flash (not FA4/B200) → prefix reuse → optional Green Contexts.
- Proven gap → L3 first-party kernel.
- For v0.2.0, the gap must come from qualified **training** profiles on both
  models. An inference-prefix or synthetic graph result cannot justify a
  training-operator speed claim.
- CUDA graph launch/replay measurements provide engine-measurement context, not an inference-policy claim.
