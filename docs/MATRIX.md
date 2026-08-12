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
python3 harness/agent_loop/run_live_metrics.py --out results/ --loops 1
MODEL=/path/to.gguf bash harness/serve/run_l1_ablation.sh
```

## Live / smoke rows (M1)

| Profile | Engine | Model | TTFT cold ms | TTFT resume ms | TPOT ms | Tool-loop ms | VRAM peak | Notes | Result |
|---------|--------|-------|--------------|----------------|---------|--------------|-----------|-------|--------|
| `live_agent_tool_loop` | ollama | `granite4.1:8b` | **7737** | **211** | **45.8** | **10005** | 14890 | #10 cold load heavy; free VRAM after ~1 GB | [json](results/live-metrics-2026-08-12/20260812T104511Z-live-tool-loop-27f8134b.json) |
| L1 cell B warm | llama-server 9190 | gemma-4 E2B Q4 | — | **25.1** (warm TTFT) | 4.55 | 311.9 | 4254 | From #16 | [json](results/l1-ablation-2026-08-12/20260812T084828Z-cell-B-fa-on-warm2-8fc4dfd7.json) |
| L1 cell A warm | llama-server 9190 | gemma-4 E2B Q4 | — | **33.1** | 4.44 | 312.8 | 4190 | FA off | [json](results/l1-ablation-2026-08-12/20260812T084823Z-cell-A-fa-off-warm1-bbad91c1.json) |
| synthetic | synthetic | n/a | — | — | — | ~profile | n/a | CI cpu path | `results/*synthetic*` |

## L1 ablation rows (#16 / RM-470) — ShipOfTheseus 2026-08-12

Host: RTX 5080 · driver 610.43.03 · CUDA 13.3 · `llama-server` 9190 · model `gemma-4-E2B-it-UD-Q4_K_XL` · profile `agent_shaped_smoke` · warm decode after load · `max_tokens=64`.

| Cell | Engine flags | TTFT ms | TPOT ms | Wall ms | VRAM peak MiB | Notes | Result |
|------|--------------|---------|---------|---------|---------------|-------|--------|
| A | `-ngl 99 -c 4096 -fa off` | **33.1** | 4.44 | 312.8 | 4190 | FA forced off | [json](results/l1-ablation-2026-08-12/20260812T084823Z-cell-A-fa-off-warm1-bbad91c1.json) |
| B | `-ngl 99 -c 4096 -fa on` | **25.1** | 4.55 | 311.9 | 4254 | FA on | [json](results/l1-ablation-2026-08-12/20260812T084828Z-cell-B-fa-on-warm2-8fc4dfd7.json) |
| C | `-fa on` + graphs=default | **22.4** | 4.35 | 296.5 | 4254 | **No graph CLI** on 9190 — not a true graphs-off baseline | [json](results/l1-ablation-2026-08-12/20260812T084829Z-cell-C-graphs-default-warm-297c2693.json) |
| D | alt quant | — | — | — | — | **Deferred**: no MXFP4/NVFP4 twin for this GGUF on host | — |

**Takeaways (honest):** On this short agent-shaped prompt + small E2B Q4 model, warm **FA on vs off** is a modest TTFT win (~25 ms vs ~33 ms); wall/TPOT nearly flat. CUDA graphs cannot be ablated independently on llama.cpp **9190** — report engine-default only. Prefer longer prefills / larger models for stronger FA signal in a follow-up.
