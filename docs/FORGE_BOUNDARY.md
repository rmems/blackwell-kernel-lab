# Forge ↔ kernel boundary contract

**Audience:** contributors adding CUDA, training code, or host GPU experiments across `rmems/*`.

## One-line rule

| Work | Lives in |
|---|---|
| **GPU kernels and engine CUDA measurements on this RTX 5080** | **`rmems/blackwell-kernel-lab`** |
| **Model training / fine-tuning / post-training ladder** | **`rmems/agoge-forger`** |

Do **not** grow a second first-party CUDA tree under the forge.

## Ownership

### This repo (`blackwell-kernel-lab`) — kernel SoT

| Layer | Meaning | Location |
|---|---|---|
| **L1** | Measure engine CUDA paths (FA, graphs, quant, prefix) | `recipes/`, `docs/`, `results/` |
| **L2** | Host scheduling (queues, Green Context notes) | docs and measured artifacts |
| **L3** | First-party `.cu` / CUTLASS | `kernels/` |

Also owns kernel documentation, measurement artifacts, and self-hosted **GPU** CI for this lab.

### Forge (`agoge-forger`) — training only

| Owns | Does **not** own |
|---|---|
| SFT / QLoRA / DPO / RL ladder | Host kernel SoT |
| Trainer multi-GPU / multi-node experience | New production `.cu` for this 5080 |
| Dataset / eval tooling for models | Duplicate L1–L3 lab under `cuda/` |

`agoge-forger/cuda/` is a **stub** — consumers point here; do not fill it with real kernels.

### Optional neuromorphic upstream

`Limen-Neural/myelin-accelerator` may be an optional dependency for niche ops. It is not the SoT for this host’s kernel lab. Target **sm_120** and do not pin ancient PTX.

## How the forge may consume kernels

Prefer contracts over FFI:

1. Measured configurations, recommended engine flags, and quant recipes.
2. Published headers, static libraries, or wheels from this repo when L3 lands.
3. Documented CLI for reproducible kernel measurements.
4. Never ad-hoc copies of `.cu` into the forge.

## Where new code goes

1. Myelin / neuromorphic ops → `Limen-Neural/myelin-accelerator`; measure or integrate on this host here.
2. Non-myelin custom CUDA, `.cu`, CUTLASS, or engine-CUDA measurement → this repo.
3. Training that uses existing engines/packages/artifacts without first-party CUDA → `agoge-forger`.

## Host rules for this RTX 5080

- **sm_120 ≠ sm_100** — no FA4 / TMEM / `tcgen05` B200 recipes for this card.
- No MIG on this consumer GPU.
- ~16 GB VRAM ceiling for kernel and engine-CUDA measurements here.
- Training multi-GPU strategy remains in the forge.

## Checklist for PRs

- [ ] New non-myelin `.cu` / CUTLASS → this repo under `kernels/` after a real L1 gap.
- [ ] New myelin / neuromorphic kernel → `Limen-Neural/myelin-accelerator`; measure/integrate here only.
- [ ] New SFT/DPO/trainer change → agoge-forger.
- [ ] Cross-repo need → artifact or doc contract, not a second CUDA tree.
- [ ] Linear: RM issues for this lab; keep GH↔Linear twins aligned.

## Related

- [KERNELS.md](KERNELS.md) — L1/L2/L3
- [MISSION.md](MISSION.md) — kernel lab mission
- Issues: #19 L3 workspace · #20 this contract · #21 L3 smoke · epic #1 / RM-175
- Forge: https://github.com/rmems/agoge-forger
