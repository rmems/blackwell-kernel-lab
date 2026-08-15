# Results schema

Harness writes JSON (and optional CSV) under `results/` (gitignored).

## Schema version

`schema_version`: **`1`**

## JSON object (one run)

```json
{
  "schema_version": 1,
  "run_id": "20260811T170000Z-synthetic-react",
  "timestamp_utc": "2026-08-11T17:00:00Z",
  "host": {
    "gpu_name": "NVIDIA GeForce RTX 5080",
    "vram_total_mb": 16303,
    "driver": "610.43.03",
    "cuda": "13.3"
  },
  "engine": {
    "name": "synthetic",
    "version": "harness-0.1",
    "endpoint": null
  },
  "model": {
    "id": null,
    "quant": null,
    "context_length": null
  },
  "workload": {
    "profile": "synthetic_react",
    "concurrency": 1,
    "steps": 5
  },
  "metrics": {
    "tool_loop_wall_ms": [120.5, 98.0],
    "ttft_ms": [],
    "tpot_ms": [],
    "tool_loop_p50_ms": 109.25,
    "ttft_p50_ms": null,
    "tpot_p50_ms": null,
    "tokens_per_s": null,
    "prefix_cache_hit_rate": null,
    "vram_peak_mb": null,
    "vram_free_mb": null
  },
  "slo": {
    "met": null,
    "notes": null
  },
  "notes": "Synthetic loop without live LLM"
}
```

## CSV (optional aggregate)

Columns: `run_id,timestamp_utc,profile,engine,model_id,concurrency,tool_loop_p50_ms,ttft_p50_ms,tpot_p50_ms,vram_peak_mb,notes`

## Metrics notes

| Field | Meaning |
|-------|---------|
| `tool_loop_p50_ms` / `ttft_p50_ms` / `tpot_p50_ms` | Median of the corresponding sample lists (emitted by harnesses) |
| `ttft_cold_ms` / `ttft_resume_ms` | Live multi-phase harness (#10): first turn vs after tool-result append |
| `ttft_cold_p50_ms` / `ttft_resume_p50_ms` | Medians of those lists |
| `vram_peak_mb` | **Sample max** (e.g. max of before/after `nvidia-smi`), not continuous peak unless a sampler is added |
| `phases[]` | Per-cycle phase breakdown (#14). One entry per completed cold→resume cycle |

## Optional fields (additive, still `schema_version: 1`)

Older runs omit these; the aggregator ignores them.

| Field | Written by | Meaning |
|-------|------------|---------|
| `workload.profile` | all | Profile name — live: `live_agent_tool_loop` / `live_coding_tool` / `live_plan_exec`; smoke: `<name>_cold_only`, `agent_shaped_smoke`, `engine_smoke` |
| `workload.max_tokens` / `workload.tool_name` | `run_live_metrics.py` | Profile decode cap and expected tool name (`null` = no tool protocol) |
| `workload.protocol_ok` | `run_live_metrics.py` | `true`/`false`, or `null` when the profile has no tool protocol |
| `model.residency` | `run_live_metrics.py` (Ollama only) | `{size_mb, size_vram_mb, resident_pct, fully_gpu_resident, context_length, quantization_level, parameter_size}` from `/api/ps`; `null` on other engines |
| `metrics.phases[]` | `run_live_metrics.py` | `{cycle, cold, cold_prefill{…}, resume_prefill{…}, tool_loop_wall_ms}`; each phase has `ttft_ms`, `wall_ms`, `decode_ms`, `tpot_ms`, `prompt_tokens`, `completion_tokens` |

`resident_pct < 100` means part of the model ran on CPU — the run is **not** a GPU kernel measurement.

Live agent path: `harness/agent_loop/run_live_metrics.py` · recipe [live-agent-metrics.md](../recipes/live-agent-metrics.md).

## Rules

- Never commit real `results/*` artifacts with secrets.  
- One file per run: `results/<run_id>.json` (`run_id` includes a short uuid suffix).  
- Aggregator only accepts `schema_version: 1` objects.  
- Bump `schema_version` only with a docs note in this file.  
