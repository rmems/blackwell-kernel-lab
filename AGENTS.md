# AGENTS.md — blackwell-kernel-lab

## What this repo is

1. **CUDA backend for [`rmems/agoge-forger`](https://github.com/rmems/agoge-forger)** on this RTX 5080: the kernels and engine CUDA paths that make local LLM fine-tuning and training honest on sm_120 / ~16 GB ([docs/KERNELS.md](docs/KERNELS.md), [docs/FORGE_CONSUME.md](docs/FORGE_CONSUME.md)).

## What this repo is not

- Not a Limen-Neural multi-repo CUDA verification suite.
- Not a dump of `cu/*.cu` copied from myelin without measurement.
- Not the training forge itself — trainer, datasets, and the SFT / QLoRA ladder live in [`rmems/agoge-forger`](https://github.com/rmems/agoge-forger).
- Not a multi-tenant cloud agent product.

## Ownership map

| Concern | Owner |
|---|---|
| Custom / host GPU kernels (L1–L3) — forge CUDA backend | **this repo** |
| SFT / QLoRA / post-training ladder | `rmems/agoge-forger` |
| Optional neuromorphic upstream | `Limen-Neural/myelin-accelerator` (dep only) |

Full rules: [docs/FORGE_BOUNDARY.md](docs/FORGE_BOUNDARY.md).

## Hardware constraints (hard)

- RTX 5080, compute **sm_120**, ~16 GB VRAM.
- Leave **≥2 GB** free when an engine measurement, training job, or desktop workload needs headroom.
- **sm_120 ≠ sm_100** — no FA4/TMEM/MIG assumptions.

## Working rules

1. Prefer measurement over micro-optimization claims without numbers.
2. Write kernel measurement outputs under `results/` (gitignored).
3. Do not commit secrets, API keys, or large model weights.
4. Linear team: **rmems (RM)**; GitHub: **rmems/blackwell-kernel-lab**. Keep GH↔Linear titles/bodies aligned when editing either side.
5. Self-hosted GPU runner work is in scope.
6. L3 kernels only after L1 leaves a proven gap (see KERNELS.md).
7. New release milestones include the version in the title; close only when that tag and release artifacts exist. The 2026-09-22 retirement of historical M1/M2/K0/CI/K1 tracks creates no retrospective releases. See [docs/RELEASE_V0_2_0.md](docs/RELEASE_V0_2_0.md).
8. Every PR: assignee **rmems**, labels, milestone, Development/project when applicable, Linear RM links. Never commit unreviewed work straight to `main`.
9. New issues originate **only in Linear**. Reuse existing GitHub mirrors; do not manually create new GitHub twins. Keep native dependencies and actual delivery evidence current.
10. v0.2.0 requires a measured training win on **both MiniCPM5 and Granite**. Freeze the operator contract from real training profiles before writing production L3 code; existing inference and synthetic benchmarks are not substitutes.

## Primary stack defaults

| Role | Default |
|---|---|
| Training / fine-tune orchestration | **agoge-forger** |
| CUDA backend / kernel SoT | **this repo** (`docs/KERNELS.md`, `docs/FORGE_CONSUME.md`) |

## Before claiming “done”

- [ ] Docs describe this repo as the agoge-forger CUDA backend, not a local-agent playground.
- [ ] VRAM/headroom implications are called out for this 16 GB host when applicable.
