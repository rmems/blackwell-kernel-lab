# Agent measurement matrix (v1)

Replaces the old multi-repo Limen verification matrix. Rows are **local agent** configurations on this RTX 5080.

## Columns

| Column | Meaning |
|--------|---------|
| Profile | Workload from [WORKLOADS.md](WORKLOADS.md) |
| Engine | ollama / llama-server / vllm / sglang / synthetic |
| Model | id + quant |
| Context | max context used |
| Concurrency | concurrent sessions |
| TTFT p50/p95 | ms |
| TPOT p50/p95 | ms |
| Tool-loop p50 | ms |
| VRAM peak | MiB |
| Notes | pass/fail / regressions |
| Result file | `results/<run_id>.json` |

## Tier-1 (freeze for M1)

| # | Profile | Engine | Model class |
|---|---------|--------|-------------|
| 1 | `synthetic_react` | synthetic harness | n/a |
| 2 | `coding_tool` | ollama **or** llama-server | 7–9B Q4/NVFP4 |
| 3 | `synthetic_plan_exec` | same as #2 | same |
| 4 | `multi_agent_n` (n=2) | same | same |

## Out of scope for v1

- Full Limen crate build matrix  
- Training / LoRA forge runs (→ **agoge-forger**)  
- Multi-node cloud serving  

## Filling the matrix

```bash
python3 harness/agent_loop/run_synthetic.py --out results/
# later: harness scripts for live engines
```

Update this table with links to result files as runs land.
