# Mission — local agent lab

## Goal

Run **local AI agents** on the RTX 5080 workstation so they are:

1. **Efficient** — low tool-loop latency, stable TTFT/TPOT, disciplined VRAM/energy use, multi-agent stability  
2. **Smarter** — reliable tools, constrained outputs, prefix/memory reuse, better concurrency behavior  

## Non-goals

| Non-goal | Owner instead |
|----------|----------------|
| Multi-repo Limen CUDA “does it build?” matrix | Retired from this epic |
| First-party CUDA kernel SoT | `myelin-accelerator` if needed later |
| **Model training / fine-tuning forge** | **`rmems/agoge-forger`** (confirmed) |
| Cloud multi-tenant agent SaaS | Out of scope |

## Success (overhaul)

- [x] README/epic describe **local agent lab**, not Limen verification  
- [ ] Backlog issues match agent-lab work; pure Limen-smoke items cancelled/parked  
- [ ] Host baseline is agent + 16 GB oriented  
- [ ] Harness runs a synthetic agent loop → JSON  
- [ ] Documented path to run a local agent on this 5080  
- [ ] Self-hosted **GPU** CI runner tracked as active work  

## Research anchors (agent efficiency)

- Agent cold prefill / resume prefill / short decode (AgentServe-style workloads)  
- NVFP4 / Q4 on consumer Blackwell under 16–32 GB  
- CUDA graphs, Flash Attention, prefix/Radix cache  
- Green Contexts for prefill/decode isolation (no MIG on consumer cards)  
