# AGENTS.md — blackwell-kernel-lab

Instructions for coding agents working in this repository.

## What this repo is

1. **GPU kernel lab (SoT)** for this RTX 5080 host: L1 engine CUDA paths, L2 scheduling, L3 first-party `.cu` / CUTLASS when measured gaps justify it (`docs/KERNELS.md`).
2. **Local agent lab** on the same host: run agents on-box, measure efficiency/smartness KPIs under **16 GB**.

## What this repo is not

- Not a Limen-Neural multi-repo CUDA verification suite  
- Not a dump of `cu/*.cu` copied from myelin without measurement  
- Not the **training forge** — that is [`rmems/agoge-forger`](https://github.com/rmems/agoge-forger)  
- Not a multi-tenant cloud agent product  

## Ownership map

| Concern | Owner |
|---------|--------|
| Custom / host GPU kernels (L1–L3) | **this repo** |
| SFT / QLoRA / post-training ladder | `rmems/agoge-forger` |
| Optional neuromorphic upstream | `Limen-Neural/myelin-accelerator` (dep only) |

## Hardware constraints (hard)

- GPU: **RTX 5080**, compute **sm_120**, **~16 GB** VRAM  
- Prefer models/quants that leave **≥2 GB** free for KV + agent overhead  
- Sweet spot: **8–14B dense** or **~20B MoE** at Q4 / NVFP4 / MXFP4  
- Full 70B interactive multi-agent is out of happy path  
- **sm_120 ≠ sm_100** — no FA4/TMEM/MIG assumptions  

## Working rules

1. Prefer **measurement** over micro-optimization claims without numbers.  
2. Write harness outputs under `results/` (gitignored); keep schema in `docs/RESULTS_SCHEMA.md`.  
3. Do not commit secrets, API keys, or large model weights.  
4. Document engine install paths for **this host** in `docs/AGENT_STACK.md`.  
5. Linear team for issues: **rmems (RM)**; GitHub: **rmems/blackwell-kernel-lab**. Keep GH↔Linear titles/bodies aligned when editing either side.  
6. Self-hosted **GPU** runner work is **in scope** (not deferred).  
7. L3 kernels only after L1 leaves a **proven** gap (see KERNELS.md).  
8. GitHub milestones include the **version in the title** (e.g. `M2 … (v0.3.0)`). Closing a milestone ⇒ tag that version. Patch bumps for in-milestone fixups only. See README “Milestones & version bumps”.  
9. Every PR: assignee **rmems**, labels, milestone, Development/project when applicable, Linear **RM-*** links. Never commit agent work straight to `main`.

## Primary stack defaults

| Role | Default |
|------|---------|
| Inference | **Ollama** and/or **llama.cpp** (`llama-server`) first |
| Optional later | vLLM / SGLang if NVFP4 + sm_120 work cleanly |
| Agent type | Local coding / tool agent over OpenAI-compatible API |
| Training | **agoge-forger** only |
| Kernel SoT | **this repo** (`docs/KERNELS.md`) |

## Before claiming “done”

- [ ] Docs match dual mission (kernels SoT + agent lab; not Limen verification)  
- [ ] Harness or recipe produces at least one JSON result when applicable  
- [ ] VRAM / model choice called out for 16 GB  
