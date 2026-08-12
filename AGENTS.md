# AGENTS.md — blackwell-kernel-lab

Instructions for coding agents working in this repository.

## What this repo is

A **local agent lab** for an RTX 5080 host: run agents on-box, measure efficiency/smartness KPIs, document the stack that actually works here.

## What this repo is not

- Not a Limen-Neural multi-repo CUDA verification suite  
- Not a second CUDA kernel tree (do not copy `cu/*.cu` from myelin)  
- Not the **training forge** — that is [`rmems/agoge-forger`](https://github.com/rmems/agoge-forger)  
- Not a multi-tenant cloud agent product  

## Hardware constraints (hard)

- GPU: **RTX 5080**, compute **sm_120**, **~16 GB** VRAM  
- Prefer models/quants that leave **≥2 GB** free for KV + agent overhead  
- Sweet spot: **8–14B dense** or **~20B MoE** at Q4 / NVFP4 / MXFP4  
- Full 70B interactive multi-agent is out of happy path  

## Working rules

1. Prefer **measurement** over micro-optimization claims without numbers.  
2. Write harness outputs under `results/` (gitignored); keep schema in `docs/RESULTS_SCHEMA.md`.  
3. Do not commit secrets, API keys, or large model weights.  
4. Document engine install paths for **this host** in `docs/AGENT_STACK.md`.  
5. Linear team for issues: **rmems (RM)**; GitHub: **rmems/blackwell-kernel-lab**. Keep GH↔Linear titles/bodies aligned when editing either side.  
6. Self-hosted **GPU** runner work is **in scope** (not deferred).  

## Primary stack defaults

| Role | Default |
|------|---------|
| Inference | **Ollama** and/or **llama.cpp** (`llama-server`) first |
| Optional later | vLLM / SGLang if NVFP4 + sm_120 work cleanly |
| Agent type | Local coding / tool agent over OpenAI-compatible API |
| Training | **agoge-forger** only |

## Before claiming “done”

- [ ] Docs match the agent-lab mission (not Limen verification)  
- [ ] Harness or recipe produces at least one JSON result when applicable  
- [ ] VRAM / model choice called out for 16 GB  
