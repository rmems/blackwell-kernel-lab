# Mission — GPU kernel + local agent lab

## Goal

On the RTX 5080 workstation:

1. **Own GPU kernel work for this host** — L1 engine CUDA paths, L2 scheduling, L3 first-party kernels when L1 proves a gap ([KERNELS.md](KERNELS.md)).  
2. Run **local AI agents** so they are:
   - **Efficient** — low tool-loop latency, stable TTFT/TPOT, disciplined VRAM/energy, multi-agent stability  
   - **Smarter** — reliable tools, constrained outputs, prefix/memory reuse, better concurrency  

## Non-goals

| Non-goal | Owner instead |
|----------|----------------|
| Multi-repo Limen CUDA “does it build?” matrix | Retired from this epic |
| **Model training / fine-tuning forge** | **`rmems/agoge-forger`** (confirmed) |
| Cloud multi-tenant agent SaaS | Out of scope |
| Blind L3 kernel writing without L1 baselines | Deferred until measured gap |

## Ownership (kernels)

| Layer | Meaning | SoT |
|-------|---------|-----|
| L1 | Measure engine CUDA (FA, graphs, quant, prefix) | **this repo** |
| L2 | Host scheduling (Green Contexts, queues) | **this repo** |
| L3 | New `.cu` / CUTLASS | **this repo** (when justified) |
| Neuromorphic optional | myelin-style ops | `Limen-Neural/myelin-accelerator` as optional dep only |

`agoge-forger/cuda/` remains a **stub**; real kernel lab work lands here.

## Success

- [x] README/epic describe dual mission (kernels SoT + agent lab), not Limen verification  
- [ ] Backlog issues match dual track; pure Limen-smoke items cancelled/parked  
- [ ] Host baseline is agent + 16 GB oriented  
- [ ] Harness runs a synthetic agent loop → JSON  
- [ ] Documented path to run a local agent on this 5080  
- [ ] L1 kernel ablation path documented and runnable (#16 / RM-470)  
- [ ] L3 workspace / smoke path exists when first first-party kernel is accepted  
- [ ] Self-hosted **GPU** CI runner tracked as active work  

## Research anchors

**Kernels (sm_120):**

- Quant fit → CUDA graphs → engine Flash (not FA4/B200) → prefix reuse → optional Green Contexts  
- Proven gap → L3 first-party kernel  

**Agents:**

- Cold prefill / resume prefill / short decode  
- NVFP4 / Q4 under 16 GB  
- Constrained tools, multi-agent TPOT  
