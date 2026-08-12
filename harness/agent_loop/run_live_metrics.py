#!/usr/bin/env python3
"""Live multi-phase agent metrics against OpenAI-compatible engines (#10 / RM-184).

Phases (AgentServe-shaped):
  1. Cold prefill  — system+tools + user  → ttft_cold_ms
  2. Short decode  — already in phase 1 wall / TPOT
  3. Resume        — tool result appended → ttft_resume_ms
  4. tool_loop     — wall from cold start through resume complete

Writes schema v1 JSON under results/. Requires a live engine (Ollama / llama-server).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Allow running as script from repo root
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from harness.serve.openai_compat import chat_completion, derive_tpot_ms  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_run_id(label: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in label)[:40]
    return f"{stamp}-{safe}-{uuid.uuid4().hex[:8]}"


def probe_gpu() -> dict:
    out = {
        "gpu_name": None,
        "vram_used_mb": None,
        "vram_free_mb": None,
        "vram_total_mb": None,
        "driver": None,
    }
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.free,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        ).strip()
        name, used, free, total, driver = [x.strip() for x in raw.split(",")]
        out["gpu_name"] = name
        out["vram_used_mb"] = int(float(used))
        out["vram_free_mb"] = int(float(free))
        out["vram_total_mb"] = int(float(total))
        out["driver"] = driver
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        pass
    return out


SYSTEM = (
    "You are a local coding agent on RTX 5080. Tools: read_file, run_shell. "
    "When you need a tool, reply with a single line: TOOL name=<tool> args=<json>. "
    "Otherwise answer briefly."
)

USER_COLD = (
    "List files in the project root using a tool call. "
    "Use: TOOL name=run_shell args={\"cmd\":\"ls\"}"
)

# Simulated tool result (no real shell) so we measure resume prefill, not tool runtime.
TOOL_RESULT = (
    "TOOL_RESULT name=run_shell ok=true\n"
    "stdout:\nAGENTS.md\nREADME.md\ndocs\nharness\nrecipes\n"
)

USER_RESUME = (
    "Tool result above. Summarize the listing in one short sentence. "
    "Do not call tools again."
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--base-url", default=os.environ.get("BKL_BASE_URL", "http://127.0.0.1:11434/v1"))
    p.add_argument("--model", default=os.environ.get("BKL_MODEL", "granite4.1:8b"))
    p.add_argument("--api-key", default=os.environ.get("BKL_API_KEY") or None)
    p.add_argument("--label", default="live-agent-metrics")
    p.add_argument("--max-tokens", type=int, default=int(os.environ.get("BKL_MAX_TOKENS", "96")))
    p.add_argument("--engine-name", default=os.environ.get("BKL_ENGINE", "openai-compat"))
    p.add_argument("--engine-version", default=os.environ.get("BKL_ENGINE_VERSION"))
    p.add_argument("--quant", default=os.environ.get("BKL_QUANT"))
    p.add_argument(
        "--loops",
        type=int,
        default=1,
        help="Repeat cold→resume cycles (median metrics when >1)",
    )
    args = p.parse_args()
    if args.loops < 1:
        print("loops must be >= 1", file=sys.stderr)
        return 2

    cold_ttfts: list[float] = []
    resume_ttfts: list[float] = []
    tpots: list[float] = []
    tool_loops: list[float] = []
    errors: list[str] = []
    last_cold_content = ""
    last_resume_content = ""

    vram_before = probe_gpu()
    loop_t0 = time.perf_counter()

    for i in range(args.loops):
        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER_COLD},
        ]
        cycle_t0 = time.perf_counter()
        cold = chat_completion(
            args.base_url,
            args.model,
            messages,
            api_key=args.api_key,
            max_tokens=args.max_tokens,
            stream=True,
        )
        if not cold["ok"]:
            errors.append(f"cold[{i}]: {cold['raw_error']}")
            break
        if isinstance(cold.get("ttft_ms"), (int, float)):
            cold_ttfts.append(float(cold["ttft_ms"]))
        tpot = derive_tpot_ms(cold.get("wall_ms"), cold.get("ttft_ms"), cold.get("completion_tokens"))
        if tpot is not None:
            tpots.append(tpot)
        last_cold_content = (cold.get("content") or "")[:200]

        messages.append({"role": "assistant", "content": cold["content"] or ""})
        messages.append(
            {
                "role": "user",
                "content": f"{TOOL_RESULT}\n\n{USER_RESUME}",
            }
        )
        resume = chat_completion(
            args.base_url,
            args.model,
            messages,
            api_key=args.api_key,
            max_tokens=args.max_tokens,
            stream=True,
        )
        if not resume["ok"]:
            errors.append(f"resume[{i}]: {resume['raw_error']}")
            break
        if isinstance(resume.get("ttft_ms"), (int, float)):
            resume_ttfts.append(float(resume["ttft_ms"]))
        tpot_r = derive_tpot_ms(
            resume.get("wall_ms"), resume.get("ttft_ms"), resume.get("completion_tokens")
        )
        if tpot_r is not None:
            tpots.append(tpot_r)
        last_resume_content = (resume.get("content") or "")[:200]
        tool_loops.append((time.perf_counter() - cycle_t0) * 1000.0)

    wall_total_ms = (time.perf_counter() - loop_t0) * 1000.0
    vram_after = probe_gpu()
    used_vals = [
        v
        for v in (vram_before.get("vram_used_mb"), vram_after.get("vram_used_mb"))
        if isinstance(v, int)
    ]
    vram_peak = max(used_vals) if used_vals else None

    ok = len(errors) == 0 and bool(cold_ttfts) and bool(tool_loops)
    rid = make_run_id(args.label)

    def p50(xs: list[float]) -> float | None:
        return float(statistics.median(xs)) if xs else None

    payload = {
        "schema_version": 1,
        "run_id": rid,
        "timestamp_utc": utc_now(),
        "host": {
            "gpu_name": vram_after.get("gpu_name") or vram_before.get("gpu_name"),
            "vram_total_mb": vram_after.get("vram_total_mb") or vram_before.get("vram_total_mb"),
            "driver": vram_after.get("driver") or vram_before.get("driver"),
            "cuda": None,
        },
        "engine": {
            "name": args.engine_name,
            "version": args.engine_version,
            "endpoint": args.base_url,
            "label": args.label,
        },
        "model": {
            "id": args.model,
            "quant": args.quant,
            "context_length": None,
        },
        "workload": {
            "profile": "live_agent_tool_loop",
            "concurrency": 1,
            "steps": args.loops,
            "stream": True,
            "phases": ["cold_prefill", "short_decode", "resume_prefill"],
        },
        "metrics": {
            "ttft_cold_ms": cold_ttfts,
            "ttft_resume_ms": resume_ttfts,
            "ttft_ms": cold_ttfts + resume_ttfts,
            "tpot_ms": tpots,
            "tool_loop_wall_ms": tool_loops,
            "ttft_cold_p50_ms": p50(cold_ttfts),
            "ttft_resume_p50_ms": p50(resume_ttfts),
            "ttft_p50_ms": p50(cold_ttfts + resume_ttfts),
            "tpot_p50_ms": p50(tpots),
            "tool_loop_p50_ms": p50(tool_loops),
            "tpot_p95_ms": (
                float(statistics.quantiles(tpots, n=20)[18]) if len(tpots) >= 2 else p50(tpots)
            ),
            "tokens_per_s": (1000.0 / p50(tpots)) if p50(tpots) and p50(tpots) > 0 else None,
            "prefix_cache_hit_rate": None,
            "vram_peak_mb": vram_peak,
            "vram_sample_note": "max(before, after) samples; not continuous peak",
            "vram_free_mb": vram_after.get("vram_free_mb"),
            "vram_before_mb": vram_before.get("vram_used_mb"),
            "vram_after_mb": vram_after.get("vram_used_mb"),
            "wall_total_ms": wall_total_ms,
        },
        "slo": {
            "met": ok,
            "notes": "ok" if ok else "; ".join(errors)[:500],
        },
        "notes": (
            f"cold_content={last_cold_content!r} resume_content={last_resume_content!r} "
            f"loops={args.loops}"
        ),
        "kernel_campaign": {
            "layer": "L1",
            "issue": "#10",
            "meaning": "Live agent-phase metrics; not custom .cu",
        },
    }

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{rid}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}")
    print(
        f"ok={ok} ttft_cold_p50={payload['metrics']['ttft_cold_p50_ms']} "
        f"ttft_resume_p50={payload['metrics']['ttft_resume_p50_ms']} "
        f"tool_loop_p50={payload['metrics']['tool_loop_p50_ms']} "
        f"tpot_p50={payload['metrics']['tpot_p50_ms']} "
        f"vram_peak={vram_peak} free={vram_after.get('vram_free_mb')}"
    )
    if errors:
        print("errors:", "; ".join(errors))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
