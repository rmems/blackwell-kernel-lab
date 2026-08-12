#!/usr/bin/env python3
"""Live multi-phase agent metrics against OpenAI-compatible engines (#10 / RM-184).

Phases (AgentServe-shaped):
  1. Cold prefill  — system+tools + user  → ttft_cold_ms (first cycle only)
  2. Short decode  — TPOT from usage when available
  3. Resume        — tool result appended → ttft_resume_ms
  4. tool_loop     — wall for each completed cold→resume cycle

Writes schema v1 JSON under results/. Requires a live engine (Ollama / llama-server).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from harness.serve.openai_compat import (  # noqa: E402
    chat_completion,
    derive_tpot_ms,
    infer_engine_name,
)


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
        # One GPU line; ignore extras if multi-GPU later.
        line = raw.splitlines()[0] if raw else ""
        parts = [x.strip() for x in line.split(",")]
        if len(parts) != 5:
            return out
        name, used, free, total, driver = parts
        out["gpu_name"] = name
        out["vram_used_mb"] = int(float(used))
        out["vram_free_mb"] = int(float(free))
        out["vram_total_mb"] = int(float(total))
        out["driver"] = driver
    except (FileNotFoundError, subprocess.SubprocessError, ValueError, IndexError):
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

TOOL_RESULT = (
    "TOOL_RESULT name=run_shell ok=true\n"
    "stdout:\nAGENTS.md\nREADME.md\ndocs\nharness\nrecipes\n"
)

USER_RESUME = (
    "Tool result above. Summarize the listing in one short sentence. "
    "Do not call tools again."
)

TOOL_RE = re.compile(r"TOOL\s+name\s*=\s*run_shell", re.IGNORECASE)


def looks_like_tool_call(text: str) -> bool:
    return bool(TOOL_RE.search(text or ""))


def p50(xs: list[float]) -> float | None:
    return float(statistics.median(xs)) if xs else None


def p95_or_none(xs: list[float]) -> float | None:
    """Empirical-ish p95 only with enough samples; else None (no exclusive extrapolate)."""
    if len(xs) < 5:
        return None
    # Inclusive nearest-rank at 95%.
    ordered = sorted(xs)
    idx = min(len(ordered) - 1, max(0, int(round(0.95 * (len(ordered) - 1)))))
    return float(ordered[idx])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--base-url", default=os.environ.get("BKL_BASE_URL", "http://127.0.0.1:11434/v1"))
    p.add_argument("--model", default=os.environ.get("BKL_MODEL", "granite4.1:8b"))
    p.add_argument("--api-key", default=os.environ.get("BKL_API_KEY") or None)
    p.add_argument("--label", default="live-agent-metrics")
    p.add_argument("--max-tokens", type=int, default=int(os.environ.get("BKL_MAX_TOKENS", "96")))
    p.add_argument(
        "--engine-name",
        default=os.environ.get("BKL_ENGINE"),
        help="Concrete engine id (ollama / llama-server). Inferred from URL if omitted.",
    )
    p.add_argument("--engine-version", default=os.environ.get("BKL_ENGINE_VERSION"))
    p.add_argument("--quant", default=os.environ.get("BKL_QUANT"))
    p.add_argument(
        "--loops",
        type=int,
        default=1,
        help="Repeat cycles. Cycle 0 = cold TTFT; later cycles recorded as warm.",
    )
    p.add_argument(
        "--require-tool-protocol",
        action="store_true",
        help="Fail cycle if cold reply lacks TOOL name=run_shell",
    )
    p.add_argument(
        "--allow-missing-tpot",
        action="store_true",
        help="Allow SLO met without TPOT (engines that omit usage); default is to fail",
    )
    args = p.parse_args()
    if args.loops < 1:
        print("loops must be >= 1", file=sys.stderr)
        return 2

    require_tpot = not args.allow_missing_tpot
    engine_name = infer_engine_name(args.base_url, args.engine_name)

    cold_ttfts: list[float] = []
    warm_ttfts: list[float] = []
    resume_ttfts: list[float] = []
    tpots: list[float] = []
    tool_loops: list[float] = []
    protocol_ok_flags: list[bool] = []
    errors: list[str] = []
    last_cold_content = ""
    last_resume_content = ""
    completed = 0

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
            if i == 0:
                cold_ttfts.append(float(cold["ttft_ms"]))
            else:
                warm_ttfts.append(float(cold["ttft_ms"]))
        tpot = derive_tpot_ms(cold.get("wall_ms"), cold.get("ttft_ms"), cold.get("completion_tokens"))
        if tpot is not None:
            tpots.append(tpot)
        last_cold_content = (cold.get("content") or "")[:200]
        proto = looks_like_tool_call(cold.get("content") or "")
        protocol_ok_flags.append(proto)
        if not proto:
            msg = f"cold[{i}]: missing TOOL name=run_shell in reply"
            if args.require_tool_protocol:
                errors.append(msg)
                break
            # Soft: still measure resume with simulated tool result, but note protocol miss.
            errors.append(msg + " (soft; continuing)")

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
        completed += 1

    wall_total_ms = (time.perf_counter() - loop_t0) * 1000.0
    vram_after = probe_gpu()
    used_vals = [
        v
        for v in (vram_before.get("vram_used_mb"), vram_after.get("vram_used_mb"))
        if isinstance(v, int)
    ]
    vram_peak = max(used_vals) if used_vals else None

    hard_errors = [e for e in errors if "(soft" not in e]
    protocol_all_ok = bool(protocol_ok_flags) and all(protocol_ok_flags)
    tpot_ok = bool(tpots) or not require_tpot
    ok = (
        not hard_errors
        and completed > 0
        and bool(cold_ttfts or warm_ttfts)
        and bool(tool_loops)
        and tpot_ok
        and (protocol_all_ok or not args.require_tool_protocol)
    )
    # Soft protocol miss still fails SLO unless all soft-only and we want honesty:
    if any("(soft" in e for e in errors) and not protocol_all_ok:
        ok = False

    rid = make_run_id(args.label)
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
            "name": engine_name,
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
            "steps": completed,
            "steps_requested": args.loops,
            "stream": True,
            "phases": ["cold_prefill", "short_decode", "resume_prefill"],
            "protocol_ok": protocol_all_ok,
        },
        "metrics": {
            "ttft_cold_ms": cold_ttfts,
            "ttft_warm_ms": warm_ttfts,
            "ttft_resume_ms": resume_ttfts,
            "ttft_ms": cold_ttfts + warm_ttfts + resume_ttfts,
            "tpot_ms": tpots,
            "tool_loop_wall_ms": tool_loops,
            "ttft_cold_p50_ms": p50(cold_ttfts),
            "ttft_warm_p50_ms": p50(warm_ttfts),
            "ttft_resume_p50_ms": p50(resume_ttfts),
            "ttft_p50_ms": p50(cold_ttfts + warm_ttfts + resume_ttfts),
            "tpot_p50_ms": p50(tpots),
            "tpot_p95_ms": p95_or_none(tpots),
            "tool_loop_p50_ms": p50(tool_loops),
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
            "notes": (
                "ok"
                if ok
                else (
                    "; ".join(errors)[:500]
                    if errors
                    else ("missing TPOT (no usage tokens)" if not tpot_ok else "incomplete")
                )
            ),
        },
        "notes": (
            f"cold_content={last_cold_content!r} resume_content={last_resume_content!r} "
            f"completed={completed}/{args.loops} protocol_ok={protocol_all_ok} "
            f"engine={engine_name} tpot_n={len(tpots)}"
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
        f"ok={ok} completed={completed}/{args.loops} "
        f"ttft_cold_p50={payload['metrics']['ttft_cold_p50_ms']} "
        f"ttft_resume_p50={payload['metrics']['ttft_resume_p50_ms']} "
        f"tool_loop_p50={payload['metrics']['tool_loop_p50_ms']} "
        f"tpot_p50={payload['metrics']['tpot_p50_ms']} "
        f"tpot_p95={payload['metrics']['tpot_p95_ms']} "
        f"vram_peak={vram_peak} free={vram_after.get('vram_free_mb')} "
        f"engine={engine_name} protocol_ok={protocol_all_ok}"
    )
    if errors:
        print("notes:", "; ".join(errors))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
