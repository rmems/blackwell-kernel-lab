# Agent workloads

Synthetic and real workloads used by the harness. Shapes follow single-GPU multi-agent research (cold / resume / short decode).

## Phases

| Phase | Description | Sensitive metric |
|-------|-------------|------------------|
| **Cold prefill** | System + tools + first user turn, uncached | TTFT |
| **Resume prefill** | Tool output or memory append with prefix reuse | TTFT of resume |
| **Short decode** | Tool call / JSON / short plan | TPOT p50/p95 |
| **Tool wall** | External tool execution (shell, HTTP, …) | Tool latency (not GPU) |
| **Loop** | Full think → tool → think cycle | Tool-loop wall time |

## Live profiles (#14 / RM-474)

Defined once in [`harness/agent_loop/profiles.py`](../harness/agent_loop/profiles.py) and consumed by both the
live harness (#10) and the L1 ablation smoke (#16), so a kernel cell and an agent row send the **same bytes**.
Prompt text is frozen: edit a profile → treat older results as a different profile.

| Profile | Cold prefill | Resume prefill | Decode | Tool protocol |
|---------|--------------|----------------|--------|----------------|
| `live_agent_tool_loop` (default) | Small system + tools blurb (~83 prompt tokens) | `run_shell` stdout | 96 tok cap, function-call shaped | `TOOL name=run_shell args={…}` |
| `live_coding_tool` | Bounded repo snippet + task (296 prompt tokens) | `read_file` content | 96 tok cap | `TOOL name=read_file args={…}` |
| `live_plan_exec` | Same snippet + planning goal (312) | `EXEC_RESULT` of step 1 | 160 tok cap, medium plan | none (`protocol_ok: null`) |

```bash
python3 harness/agent_loop/run_live_metrics.py --out results/ --profile live_coding_tool --loops 3
python3 harness/serve/smoke_openai.py --out results/ --stream --skip-content-slo \
  --profile live_coding_tool --label cell-B-fa-on     # #16: cold prefill only
```

`--profile` on the smoke path records `workload.profile = "<name>_cold_only"` — one turn, no resume phase.

### Per-phase output

Each completed cold→resume cycle is written to `metrics.phases[]`:

```json
{"cycle": 1, "cold": false,
 "cold_prefill":   {"ttft_ms": 78.4, "wall_ms": 166.5, "decode_ms": 88.1, "tpot_ms": 7.34,
                    "prompt_tokens": 83, "completion_tokens": 13},
 "resume_prefill": {"ttft_ms": 79.7, "wall_ms": 322.1, "decode_ms": 242.5, "tpot_ms": 7.35,
                    "prompt_tokens": 151, "completion_tokens": 34},
 "tool_loop_wall_ms": 488.9}
```

Cycle 0 pays model load and has no prefix cache; later cycles are warm. `cold` marks which is which.

### Measured — 2026-08-15, `granite4.1:8b-ctx8k` Q4_K_M, 8192 ctx, 100% GPU

| Profile | Cold-turn TTFT p50 ms | Resume TTFT p50 ms | TPOT p50 ms | Tool-loop p50 ms | Result |
|---------|-----------------------|--------------------|-------------|------------------|--------|
| `live_agent_tool_loop` | 2440 (cycle 0 incl. load; 78.4 warm) | 76.7 | 7.35 | 489 | [json](results/cookbook-2026-08-15/20260815T130312Z-cookbook-granite-ctx8k-tool-loop-f8cdbad8.json) |
| `live_coding_tool` | 109.7 | 86.5 | 7.35 | 331 | [json](results/cookbook-2026-08-15/20260815T130321Z-cookbook-granite-ctx8k-coding-tool-9bd9069d.json) |
| `live_plan_exec` | 130.1 | 82.5 | 7.39 | 1718 | [json](results/cookbook-2026-08-15/20260815T130324Z-cookbook-granite-ctx8k-plan-exec-508725f6.json) |

Honest caveats: the `live_coding_tool` / `live_plan_exec` runs reused an already-resident model, so their cycle-0
TTFT is **cold prompt, warm weights** — comparable to each other, not to the 2440 ms load figure.
`live_plan_exec`'s tool-loop is long because it decodes a 160-token plan, not because prefill is slower.
VRAM and residency for all three: [MODELS.md](MODELS.md).

## Profiles (synthetic — no live engine)

Run by `harness/agent_loop/run_synthetic.py` (`--profile`), used by CPU CI.

### `synthetic_react`

Short ReAct-style steps: many resume prefills, very short decodes (function-call shaped).

### `synthetic_plan_exec`

Longer cold prefill (plan), medium decode, fewer tools.

### `coding_tool`

Coding agent: repo context snippets + tools (`read`, `run`, `edit` proxies). Prefill-heavy if prompts include large files — keep snippets bounded for 16 GB.

### `multi_agent_n`

N concurrent synthetic sessions sharing one GPU. Track per-session TTFT/TPOT and SLO % meeting thresholds.

## SLO templates (starting points — calibrate per model)

| Metric | Starting target (interactive) |
|--------|-------------------------------|
| TTFT p50 | &lt; 1.0 s (short prompts) |
| TPOT p50 | stable emission; flag p95 spikes |
| Tool-loop | task-dependent; always record wall ms |

Measured reference (2026-08-15, `granite4.1:8b-ctx8k` Q4_K_M, 8192 ctx, 100% GPU): warm TTFT 78–130 ms,
first-load TTFT 2.4 s, TPOT p50 7.35 ms / p95 7.76 ms, tool-loop 331–1718 ms depending on profile.

Calibrate SLOs after first baseline matrix fill; store thresholds in result JSON `slo` field when used.
A row only counts as a GPU measurement if `model.residency.resident_pct` is 100 — see [MODELS.md](MODELS.md).
