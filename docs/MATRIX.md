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

## Live multi-phase rows (M1 / #10)

Columns match cold → tool → resume. Do **not** put single-turn L1 smokes here.

| Profile | Engine | Model | TTFT cold ms | TTFT resume ms | TPOT p50 ms | Tool-loop p50 ms | VRAM peak | Notes | Result |
|---------|--------|-------|--------------|----------------|-------------|------------------|-----------|-------|--------|
| `live_agent_tool_loop` | ollama | `granite4.1:8b` | **7737** | **211** | **45.8** | **10005** | 14890 | cold load heavy; free after ~1 GB; TPOT p50 aggregates cold+resume samples; historical fixture (protocol not re-validated). **2026-08-15 root cause: that run was only 47% GPU-resident** (ctx unpinned) — see cookbook rows below | [fixture](results/live-metrics-2026-08-12/20260812T104511Z-live-tool-loop-27f8134b.json) |

## Cookbook + live profile rows (#13 / #14) — 2026-08-15

Host: RTX 5080 · driver 610.43.03 · Ollama 0.30.6 · VRAM peak is whole-board (~2.0 GB desktop included) ·
`resident` from `model.residency` (`/api/ps`). Context: [MODELS.md](MODELS.md) · [WORKLOADS.md](WORKLOADS.md).

| Profile | Model (ctx) | Quant | resident | TTFT cold ms | TTFT resume ms | TPOT p50 ms | Tool-loop p50 ms | VRAM peak | Notes | Result |
|---------|-------------|-------|----------|--------------|----------------|-------------|------------------|-----------|-------|--------|
| `live_agent_tool_loop` | `granite4.1:8b-ctx8k` (8192) | Q4_K_M | **100%** | 2440 | **76.7** | **7.35** | **489** | 8748 | cold includes model load; warm cold-turn 78.4 | [json](results/cookbook-2026-08-15/20260815T130312Z-cookbook-granite-ctx8k-tool-loop-f8cdbad8.json) |
| `live_agent_tool_loop` | `granite4.1:8b` (131072) | Q4_K_M | **47%** | 6260 | 205 | 45.1 | 2361 | 14918 | same weights, ctx unpinned → 53% CPU offload | [json](results/cookbook-2026-08-15/20260815T130230Z-cookbook-granite-tool-loop-26deb30b.json) |
| `live_coding_tool` | `granite4.1:8b-ctx8k` (8192) | Q4_K_M | 100% | 109.7 | 86.5 | 7.35 | 331 | 8752 | 296-token cold prefill; warm weights | [json](results/cookbook-2026-08-15/20260815T130321Z-cookbook-granite-ctx8k-coding-tool-9bd9069d.json) |
| `live_plan_exec` | `granite4.1:8b-ctx8k` (8192) | Q4_K_M | 100% | 130.1 | 82.5 | 7.39 | 1718 | 8752 | 160-token plan decode; no tool protocol | [json](results/cookbook-2026-08-15/20260815T130324Z-cookbook-granite-ctx8k-plan-exec-508725f6.json) |
| `live_agent_tool_loop` | `phi4:14b-ctx8k` (8192) | Q4_K_M | 100% | 3635 | 80.7 | 11.6 | 3188 | 12460 | 14.7B dense fits fully at 8k | [json](results/cookbook-2026-08-15/20260815T130419Z-cookbook-phi4-14b-ctx8k-a578a712.json) |
| `live_agent_tool_loop` | `qwen3.5:9b-q8_0-ctx8k` (8192) | Q8_0 | 100% | 5070 | 1571 | 2.51 ⚠ | 4900 | 12344 | thinking model: TTFT absorbs reasoning, TPOT is an artifact; needed `--max-tokens 512` | [json](results/cookbook-2026-08-15/20260815T130410Z-cookbook-qwen9b-q8-ctx8k-512-79ba21f4.json) |

Cold TTFT is comparable **only** within a load state: rows 1/2/5/6 paid a model load, rows 3/4 reused a resident model.

## Warm single-turn L1 rows (#16) — separate columns

| Profile | Engine | Model | Warm TTFT ms | Request wall ms | TPOT ms | VRAM peak | Notes | Result |
|---------|--------|-------|--------------|-----------------|---------|-----------|-------|--------|
| `agent_shaped_smoke` cell B | llama-server 9190 | gemma-4 E2B Q4 | **25.1** | 311.9 | 4.55 | 4254 | FA on; not a resume/tool-loop | [json](results/l1-ablation-2026-08-12/20260812T084828Z-cell-B-fa-on-warm2-8fc4dfd7.json) |
| `agent_shaped_smoke` cell A | llama-server 9190 | gemma-4 E2B Q4 | **33.1** | 312.8 | 4.44 | 4190 | FA off | [json](results/l1-ablation-2026-08-12/20260812T084823Z-cell-A-fa-off-warm1-bbad91c1.json) |
| synthetic | synthetic | n/a | — | ~profile | — | n/a | CI cpu path | `results/*synthetic*` |

## L1 ablation rows (#16 / RM-470) — ShipOfTheseus 2026-08-12

Host: RTX 5080 · driver 610.43.03 · CUDA 13.3 · `llama-server` 9190 · model `gemma-4-E2B-it-UD-Q4_K_XL` · profile `agent_shaped_smoke` · warm decode after load · `max_tokens=64`.

| Cell | Engine flags | TTFT ms | TPOT ms | Wall ms | VRAM peak MiB | Notes | Result |
|------|--------------|---------|---------|---------|---------------|-------|--------|
| A | `-ngl 99 -c 4096 -fa off` | **33.1** | 4.44 | 312.8 | 4190 | FA forced off | [json](results/l1-ablation-2026-08-12/20260812T084823Z-cell-A-fa-off-warm1-bbad91c1.json) |
| B | `-ngl 99 -c 4096 -fa on` | **25.1** | 4.55 | 311.9 | 4254 | FA on | [json](results/l1-ablation-2026-08-12/20260812T084828Z-cell-B-fa-on-warm2-8fc4dfd7.json) |
| C | `-fa on` + graphs=default | **22.4** | 4.35 | 296.5 | 4254 | **No graph CLI** on 9190 — not a true graphs-off baseline | [json](results/l1-ablation-2026-08-12/20260812T084829Z-cell-C-graphs-default-warm-297c2693.json) |
| D | alt quant | — | — | — | — | **Deferred**: no MXFP4/NVFP4 twin for this GGUF on host | — |

**Takeaways (honest):** On this short agent-shaped prompt + small E2B Q4 model, warm **FA on vs off** is a modest TTFT win (~25 ms vs ~33 ms); wall/TPOT nearly flat. CUDA graphs cannot be ablated independently on llama.cpp **9190** — report engine-default only. Prefer longer prefills / larger models for stronger FA signal in a follow-up.
