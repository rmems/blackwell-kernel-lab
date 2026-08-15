"""Named agent workload profiles (#14 / RM-474).

One place to define the AgentServe-shaped phases documented in
[docs/WORKLOADS.md](../../docs/WORKLOADS.md):

  cold prefill  → short structured decode → resume prefill → short decode

Consumers:
  * `harness/agent_loop/run_live_metrics.py` (#10) — full multi-step loop
  * `harness/serve/smoke_openai.py` (#16) — single-turn cold prefill only

Prompt text is fixed on purpose: cells are only comparable if the workload is
byte-identical across engines, flags and quants. Change a profile → treat old
results as a different profile.
"""

from __future__ import annotations

import re

# Bounded repo snippet for prefill-heavy profiles. Inline (not read from disk)
# so the workload is identical on every host and after every refactor.
_SNIPPET = """\
# --- context: harness/report/aggregate.py (excerpt) ---
COLUMNS = [
    "run_id", "timestamp_utc", "profile", "engine", "model_id",
    "concurrency", "tool_loop_p50_ms", "ttft_p50_ms", "tpot_p50_ms",
    "vram_peak_mb", "notes",
]
SUPPORTED_SCHEMA = 1

def row_from(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != SUPPORTED_SCHEMA:
        raise ValueError("unsupported schema_version")
    metrics = data["metrics"]
    workload = data.get("workload") or {}
    return {
        "run_id": data.get("run_id"),
        "profile": workload.get("profile"),
        "tool_loop_p50_ms": metrics.get("tool_loop_p50_ms"),
        "vram_peak_mb": metrics.get("vram_peak_mb"),
    }
# --- end context ---"""

PROFILES: dict[str, dict] = {
    # Default: unchanged text from the 2026-08-12 live fixture so historical
    # MATRIX rows stay comparable. Do not edit these four strings.
    "live_agent_tool_loop": {
        "description": "Short ReAct tool loop: small cold prefill, one tool result, short decode.",
        "tool_name": "run_shell",
        "max_tokens": 96,
        "system": (
            "You are a local coding agent on RTX 5080. Tools: read_file, run_shell. "
            "When you need a tool, reply with a single line: TOOL name=<tool> args=<json>. "
            "Otherwise answer briefly."
        ),
        "user_cold": (
            "List files in the project root using a tool call. "
            "Use: TOOL name=run_shell args={\"cmd\":\"ls\"}"
        ),
        "tool_result": (
            "TOOL_RESULT name=run_shell ok=true\n"
            "stdout:\nAGENTS.md\nREADME.md\ndocs\nharness\nrecipes\n"
        ),
        "user_resume": (
            "Tool result above. Summarize the listing in one short sentence. "
            "Do not call tools again."
        ),
    },
    # Prefill-heavy: repo snippet in the cold turn, file read on resume.
    "live_coding_tool": {
        "description": "Coding agent: bounded repo snippet in cold prefill, read_file result on resume.",
        "tool_name": "read_file",
        "max_tokens": 96,
        "system": (
            "You are a local coding agent on RTX 5080. Tools: read_file, run_shell. "
            "When you need a tool, reply with a single line: TOOL name=<tool> args=<json>. "
            "Otherwise answer briefly."
        ),
        "user_cold": (
            f"{_SNIPPET}\n\n"
            "The CSV aggregator above drops rows silently. Open the schema doc before "
            "changing anything. Reply with exactly one line: "
            "TOOL name=read_file args={\"path\":\"docs/RESULTS_SCHEMA.md\"}"
        ),
        "tool_result": (
            "TOOL_RESULT name=read_file ok=true\n"
            "path: docs/RESULTS_SCHEMA.md\n"
            "content:\n"
            "# Results schema\n"
            "schema_version: 1\n"
            "metrics: tool_loop_wall_ms, ttft_ms, tpot_ms, vram_peak_mb\n"
            "Rules: one file per run; aggregator only accepts schema_version 1.\n"
        ),
        "user_resume": (
            "Tool result above. In one short sentence, say which schema_version the "
            "aggregator accepts. Do not call tools again."
        ),
    },
    # Plan-then-execute: long cold prefill, medium decode, no tool protocol.
    "live_plan_exec": {
        "description": "Plan/execute: long cold prefill, medium decode, no tool-call protocol.",
        "tool_name": None,
        "max_tokens": 160,
        "system": (
            "You are a local planning agent on RTX 5080 (16 GB). You write short, "
            "numbered plans for measurement tasks. Be concise and concrete."
        ),
        "user_cold": (
            f"{_SNIPPET}\n\n"
            "Goal: measure whether Flash Attention changes time-to-first-token for a "
            "7-9B Q4 model on a single 16 GB GPU, using the aggregator above to collect "
            "results. Constraints: one GPU, no new model downloads, results must be "
            "reproducible from committed JSON. Write a numbered plan of at most 5 steps."
        ),
        "tool_result": (
            "EXEC_RESULT ok=true\n"
            "step 1 done: llama-server started with -fa off, 4096 ctx, all layers on GPU.\n"
            "ttft_ms=33.1 vram_peak_mb=4190\n"
        ),
        "user_resume": (
            "Step 1 result above. Give the next step only, in one short sentence."
        ),
    },
}

DEFAULT_PROFILE = "live_agent_tool_loop"


def get(name: str) -> dict:
    """Return a profile dict; raises KeyError with the valid names listed."""
    try:
        return PROFILES[name]
    except KeyError:
        raise KeyError(f"unknown profile {name!r}; known: {', '.join(sorted(PROFILES))}") from None


def names() -> list[str]:
    return sorted(PROFILES)


def cold_prompt(name: str) -> str:
    """Single-turn cold-prefill prompt (system + first user turn) for #16 smokes."""
    p = get(name)
    return f"{p['system']}\n\n{p['user_cold']}"


def tool_line_re(tool_name: str) -> re.Pattern[str]:
    """Full single-line protocol: TOOL name=<tool> args={...}.

    Rejects prose ("I cannot issue TOOL name=...") and name=<tool>_extra.
    """
    return re.compile(
        rf"^\s*TOOL\s+name\s*=\s*{re.escape(tool_name)}\s+args\s*=\s*\{{.+\}}\s*$",
        re.IGNORECASE,
    )
