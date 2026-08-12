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

## Profiles

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

Calibrate SLOs after first baseline matrix fill; store thresholds in result JSON `slo` field when used.
