#!/usr/bin/env python3
"""Live multi-phase agent metrics against OpenAI-compatible engines (#10 / RM-184).

Phases (AgentServe-shaped):
  1. Cold prefill  — system+tools + user  → ttft_cold_ms (first cycle only)
  2. Short decode  — TPOT from usage when available
  3. Resume        — tool result appended → ttft_resume_ms
  4. tool_loop     — wall for each completed cold→resume cycle

Workload text comes from a named profile (`--profile`, see
`harness/agent_loop/profiles.py` / docs/WORKLOADS.md). Per-cycle phase timings are
written to `metrics.phases`.

Writes schema v1 JSON under results/. Requires a live engine (Ollama / llama-server).
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from harness.agent_loop import profiles  # noqa: E402
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


def _ollama_name_matches(requested: str, candidate: object) -> bool:
    """Ollama /api/ps often reports name as tag:latest; callers usually omit :latest."""
    if not isinstance(candidate, str) or not requested:
        return False
    req = requested.strip()
    cand = candidate.strip()
    if req == cand:
        return True

    def _strip_latest(tag: str) -> str:
        return tag[: -len(":latest")] if tag.endswith(":latest") else tag

    return _strip_latest(req) == _strip_latest(cand)


def probe_ollama_residency(base_url: str, model: str) -> dict | None:
    """Ollama-only: /api/ps → how much of the loaded model actually sits in VRAM.

    Returns None for other engines / when the endpoint is unavailable. This is the
    machine-checkable evidence behind "fully GPU-resident" claims (#13).
    """
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    try:
        with urllib.request.urlopen(root + "/api/ps", timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (
        urllib.error.URLError,
        TimeoutError,
        socket.timeout,
        json.JSONDecodeError,
        OSError,
        ValueError,
    ):
        return None
    if not isinstance(data, dict):
        return None
    entries = data.get("models")
    if not isinstance(entries, list):
        return None
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if not (
            _ollama_name_matches(model, entry.get("name"))
            or _ollama_name_matches(model, entry.get("model"))
        ):
            continue
        size = entry.get("size")
        size_vram = entry.get("size_vram")
        if not isinstance(size, int) or not isinstance(size_vram, int) or size <= 0:
            return None
        pct = round(100.0 * size_vram / size, 2)
        details = entry.get("details") if isinstance(entry.get("details"), dict) else {}
        return {
            "source": "ollama /api/ps",
            "size_mb": round(size / (1024 * 1024)),
            "size_vram_mb": round(size_vram / (1024 * 1024)),
            "resident_pct": pct,
            "fully_gpu_resident": size_vram == size,
            "context_length": entry.get("context_length"),
            "quantization_level": details.get("quantization_level"),
            "parameter_size": details.get("parameter_size"),
        }
    return None


def looks_like_tool_call(text: str, pattern) -> bool:
    for line in (text or "").splitlines():
        if pattern.match(line.strip()):
            return True
    return False


def phase_record(result: dict, tpot: float | None) -> dict:
    """One phase (cold or resume) as prefill + decode timings."""
    ttft = result.get("ttft_ms")
    wall = result.get("wall_ms")
    decode_ms = None
    if isinstance(ttft, (int, float)) and isinstance(wall, (int, float)):
        decode_ms = max(float(wall) - float(ttft), 0.0)
    return {
        "ttft_ms": ttft,
        "wall_ms": wall,
        "decode_ms": decode_ms,
        "tpot_ms": tpot,
        "prompt_tokens": result.get("prompt_tokens"),
        "completion_tokens": result.get("completion_tokens"),
    }


def p50(xs: list[float]) -> float | None:
    return float(statistics.median(xs)) if xs else None


def p95_or_none(xs: list[float]) -> float | None:
    """Nearest-rank p95 only with enough samples; else None (no exclusive extrapolate)."""
    if len(xs) < 5:
        return None
    ordered = sorted(xs)
    # nearest-rank: ceil(0.95 * n) - 1  ==  (95*n + 99)//100 - 1
    idx = (95 * len(ordered) + 99) // 100 - 1
    idx = min(max(idx, 0), len(ordered) - 1)
    return float(ordered[idx])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--base-url", default=os.environ.get("BKL_BASE_URL", "http://127.0.0.1:11434/v1"))
    p.add_argument("--model", default=os.environ.get("BKL_MODEL", "granite4.1:8b"))
    p.add_argument("--api-key", default=os.environ.get("BKL_API_KEY") or None)
    p.add_argument("--label", default="live-agent-metrics")
    p.add_argument(
        "--profile",
        default=os.environ.get("BKL_PROFILE", profiles.DEFAULT_PROFILE),
        choices=profiles.names(),
        help="Workload profile (docs/WORKLOADS.md); fixes prompts and max_tokens",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Override the profile's max_tokens (env BKL_MAX_TOKENS)",
    )
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
        help="Fail cycle if the cold reply lacks the profile's TOOL line (profiles with a tool only)",
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

    profile = profiles.get(args.profile)
    tool_name = profile["tool_name"]
    tool_re = profiles.tool_line_re(tool_name) if tool_name else None
    env_max_tokens = os.environ.get("BKL_MAX_TOKENS")
    if args.max_tokens is not None:
        max_tokens = args.max_tokens
    elif env_max_tokens:
        try:
            max_tokens = int(env_max_tokens)
        except ValueError:
            p.error(f"BKL_MAX_TOKENS is not an int: {env_max_tokens!r}")
    else:
        max_tokens = profile["max_tokens"]
    if max_tokens < 1:
        p.error("--max-tokens must be >= 1")

    require_tpot = not args.allow_missing_tpot
    engine_name = infer_engine_name(args.base_url, args.engine_name)

    cold_ttfts: list[float] = []
    warm_ttfts: list[float] = []
    resume_ttfts: list[float] = []
    tpots: list[float] = []
    tool_loops: list[float] = []
    protocol_ok_flags: list[bool] = []
    phases: list[dict] = []
    errors: list[str] = []
    last_cold_content = ""
    last_resume_content = ""
    completed = 0
    residency = None

    vram_before = probe_gpu()
    loop_t0 = time.perf_counter()

    for i in range(args.loops):
        messages: list[dict[str, str]] = [
            {"role": "system", "content": profile["system"]},
            {"role": "user", "content": profile["user_cold"]},
        ]
        cycle_t0 = time.perf_counter()
        cold = chat_completion(
            args.base_url,
            args.model,
            messages,
            api_key=args.api_key,
            max_tokens=max_tokens,
            stream=True,
        )
        if not cold["ok"]:
            errors.append(f"cold[{i}]: {cold['raw_error']}")
            break
        if residency is None and engine_name == "ollama":
            # Probe while the model is loaded (cycle 0 has just paid the load cost).
            residency = probe_ollama_residency(args.base_url, args.model)
        if isinstance(cold.get("ttft_ms"), (int, float)):
            if i == 0:
                cold_ttfts.append(float(cold["ttft_ms"]))
            else:
                warm_ttfts.append(float(cold["ttft_ms"]))
        tpot = derive_tpot_ms(cold.get("wall_ms"), cold.get("ttft_ms"), cold.get("completion_tokens"))
        if tpot is not None:
            tpots.append(tpot)
        last_cold_content = (cold.get("content") or "")[:200]
        if tool_re is not None:
            proto = looks_like_tool_call(cold.get("content") or "", tool_re)
            protocol_ok_flags.append(proto)
            if not proto:
                msg = f"cold[{i}]: missing TOOL name={tool_name} in reply"
                if args.require_tool_protocol:
                    errors.append(msg)
                    break
                # Soft: still measure resume with simulated tool result, but note protocol miss.
                errors.append(msg + " (soft; continuing)")

        messages.append({"role": "assistant", "content": cold["content"] or ""})
        messages.append(
            {
                "role": "user",
                "content": f"{profile['tool_result']}\n\n{profile['user_resume']}",
            }
        )
        resume = chat_completion(
            args.base_url,
            args.model,
            messages,
            api_key=args.api_key,
            max_tokens=max_tokens,
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
        cycle_wall = (time.perf_counter() - cycle_t0) * 1000.0
        tool_loops.append(cycle_wall)
        phases.append(
            {
                "cycle": i,
                "cold": i == 0,
                # Cycle 0 pays model load + no prefix cache; later cycles are warm.
                "cold_prefill": phase_record(cold, tpot),
                "resume_prefill": phase_record(resume, tpot_r),
                "tool_loop_wall_ms": cycle_wall,
            }
        )
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
    # None = profile has no tool protocol to check (e.g. live_plan_exec).
    protocol_all_ok = (
        (bool(protocol_ok_flags) and all(protocol_ok_flags)) if tool_re is not None else None
    )
    protocol_gate_ok = protocol_all_ok is not False
    tpot_ok = bool(tpots) or not require_tpot
    ok = (
        not hard_errors
        and completed > 0
        and bool(cold_ttfts or warm_ttfts)
        and bool(tool_loops)
        and tpot_ok
        and (protocol_gate_ok or not args.require_tool_protocol)
    )
    # Soft protocol miss still fails SLO unless all soft-only and we want honesty:
    if any("(soft" in e for e in errors) and not protocol_gate_ok:
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
            "quant": args.quant or (residency or {}).get("quantization_level"),
            "context_length": (residency or {}).get("context_length"),
            # Engine-reported weight residency (ollama only); None elsewhere.
            "residency": residency,
        },
        "workload": {
            "profile": args.profile,
            "profile_description": profile["description"],
            "concurrency": 1,
            "steps": completed,
            "steps_requested": args.loops,
            "stream": True,
            "max_tokens": max_tokens,
            "phases": ["cold_prefill", "short_decode", "resume_prefill"],
            "tool_name": tool_name,
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
            # Aggregate ttft_p50 is cold-only so CSV does not mix cold+resume phases.
            "ttft_p50_ms": p50(cold_ttfts),
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
            # Per-cycle phase breakdown (#14): cold prefill / resume prefill /
            # short decode, one entry per completed cold→resume cycle.
            "phases": phases,
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
            f"profile={args.profile} cold_content={last_cold_content!r} "
            f"resume_content={last_resume_content!r} "
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
        f"ok={ok} profile={args.profile} completed={completed}/{args.loops} "
        f"ttft_cold_p50={payload['metrics']['ttft_cold_p50_ms']} "
        f"ttft_resume_p50={payload['metrics']['ttft_resume_p50_ms']} "
        f"tool_loop_p50={payload['metrics']['tool_loop_p50_ms']} "
        f"tpot_p50={payload['metrics']['tpot_p50_ms']} "
        f"tpot_p95={payload['metrics']['tpot_p95_ms']} "
        f"vram_peak={vram_peak} free={vram_after.get('vram_free_mb')} "
        f"engine={engine_name} protocol_ok={protocol_all_ok} "
        f"resident={(residency or {}).get('resident_pct')}%"
    )
    if errors:
        print("notes:", "; ".join(errors))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
