# Forge ↔ kernel boundary contract

**Audience:** agents and humans adding CUDA, training code, or host GPU experiments across `rmems/*`.

## One-line rule

| Work | Lives in |
|------|----------|
| **GPU kernels + agent measurement on this RTX 5080** | **`rmems/blackwell-kernel-lab`** (this repo) |
| **Model training / fine-tuning / post-training ladder** | **`rmems/agoge-forger`** |

Do **not** grow a second first-party CUDA tree under the forge.

## Ownership

### This repo (`blackwell-kernel-lab`) — kernel SoT

| Layer | Meaning | Location |
|-------|---------|----------|
| **L1** | Measure engine CUDA paths (FA, graphs, quant, prefix) | `harness/`, `recipes/`, `docs/`, `results/` |
| **L2** | Host scheduling (queues, Green Context notes) | docs + harness when landed |
| **L3** | First-party `.cu` / CUTLASS | `kernels/` (see issue #19) |

Also owns: local agent stack docs, measurement schema, matrix, and self-hosted **GPU** CI for this lab.

### Forge (`agoge-forger`) — training only

| Owns | Does **not** own |
|------|------------------|
| SFT / QLoRA / DPO / RL ladder | Host kernel SoT |
| Trainer multi-GPU / multi-node experience | New production `.cu` for this 5080 |
| Dataset / eval harness for models | Duplicate L1–L3 lab under `cuda/` |

`agoge-forger/cuda/` (if present) is a **stub** — point consumers here; do not fill it with real kernels.

### Optional neuromorphic upstream

`Limen-Neural/myelin-accelerator` may be an **optional dependency** for niche ops. It is **not** the SoT for this host’s kernel lab.

When those ops are **run or measured on this host**, follow [HOST_BASELINE.md](HOST_BASELINE.md): target **`sm_120`**, and do **not** pin ancient PTX (avoids `InvalidPtx` / load failures on the 5080). Upstream myelin may support other arches; this host’s integration path stays sm_120-clean.

## How the forge may consume kernels

Prefer **contracts over FFI**:

1. **Artifacts** — measured configs, recommended engine flags, quant recipes, MATRIX rows.  
2. **File / package** — published headers, static libs, or wheels from **this** repo when L3 lands.  
3. **Documented CLI** — engine flags and smoke scripts agents can re-run.  
4. **Avoid** ad-hoc copies of `.cu` into the forge “for convenience.”

If training needs a **non-myelin** custom CUDA op: implement or wrap it in **blackwell-kernel-lab**, then consume the artifact from agoge-forger. Neuromorphic / myelin ops stay in `myelin-accelerator` (see decision tree).

## Where new code goes (decision tree)

Evaluate **top to bottom**. First matching branch wins.

1. **Myelin / neuromorphic upstream ops take precedence** over the generic “kernel on this 5080” branch — implement or extend in `Limen-Neural/myelin-accelerator`; measure/integrate on this host here (sm_120 rules in [HOST_BASELINE.md](HOST_BASELINE.md)).
2. **Non-myelin custom CUDA wins over training.** A training feature that needs a new `.cu`, CUTLASS kernel, or custom CUDA op is **not** an excuse to open a CUDA tree under the forge — implement or wrap the op **here** first, then consume it from `agoge-forger` via artifacts/packages/CLI.

```text
Is it neuromorphic / myelin-class ops (SNN, niche myelin kernels)?
  → Limen-Neural/myelin-accelerator (optional dep)
  → still measure / integrate on this host in blackwell-kernel-lab (sm_120; no ancient PTX)

Is it measuring or writing non-myelin GPU kernels / agent VRAM-latency on this 5080?
OR does a training feature need a new custom CUDA / .cu / CUTLASS op (not myelin)?
  → blackwell-kernel-lab   (kernel SoT; no forge .cu tree)

Is it training / fine-tune / preference optimization that only uses
existing engines, packages, or artifacts (no new first-party CUDA)?
  → agoge-forger
```

## Host rules for **this** RTX 5080 (not global forge policy)

These apply to **work targeting this machine** (this lab, and any forge job that **runs or measures on ShipOfTheseus / RTX 5080**). They do **not** ban other GPUs (e.g. B200) in `agoge-forger` training recipes elsewhere.

- On this host: **sm_120 ≠ sm_100** — no FA4 / TMEM / `tcgen05` B200 recipes **for this card**.  
- **No MIG** on this consumer card.  
- **~16 GB** VRAM ceiling for agent + kernel experiments **here**.  
- Training multi-GPU strategy lives in the **forge**; kernel SoT for this host still does not move into `agoge-forger/cuda/`.

## Checklist for PRs

- [ ] New `.cu` / CUTLASS → this repo under `kernels/` (after L1 gap is real).  
- [ ] New SFT/DPO/trainer change → agoge-forger.  
- [ ] Cross-repo need → artifact or doc contract, not a second CUDA tree.  
- [ ] Linear: **RM** issues for this lab; keep GH↔Linear twins aligned.

## Related

- [KERNELS.md](KERNELS.md) — L1/L2/L3  
- [MISSION.md](MISSION.md) — dual track  
- [AGENT_STACK.md](AGENT_STACK.md) — local engines  
- Issues: #19 L3 workspace · #20 this contract · #21 L3 smoke · epic #1 / RM-175  
- Forge: https://github.com/rmems/agoge-forger  
