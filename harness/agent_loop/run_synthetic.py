#!/usr/bin/env python3
"""Synthetic agent-loop harness (no live LLM required).

Emits schema_version=1 JSON under results/ for cold/resume/decode-shaped steps.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_id(profile: str) -> str:
    # Second precision + short uuid so concurrent/same-second runs never collide.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{profile}-{uuid.uuid4().hex[:8]}"


def probe_gpu() -> dict:
    host = {
        "gpu_name": None,
        "vram_total_mb": None,
        "driver": None,
        "cuda": None,
    }
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        ).strip()
        if out:
            name, vram, driver = [p.strip() for p in out.split(",")]
            host["gpu_name"] = name
            host["vram_total_mb"] = int(float(vram))
            host["driver"] = driver
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        pass
    try:
        nvcc = subprocess.check_output(
            ["nvcc", "--version"], text=True, timeout=10
        )
        for line in nvcc.splitlines():
            if "release" in line.lower():
                parts = line.split("release")
                if len(parts) > 1:
                    host["cuda"] = parts[1].split(",")[0].strip()
                break
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    return host


def _profile_scale(profile: str) -> float:
    """Distinct synthetic cost shapes per profile (not label-only)."""
    if profile == "synthetic_plan_exec":
        return 1.6  # heavier cold prefill
    if profile == "coding_tool":
        return 1.25  # medium tool/result size
    return 1.0  # synthetic_react


def synthetic_loop(steps: int, base_ms: float, profile: str) -> dict:
    """Simulate think→tool→think with cold first step + short resumes."""
    if steps < 1:
        raise ValueError("steps must be >= 1")

    scale = _profile_scale(profile)
    tool_loop_ms: list[float] = []
    ttft_ms: list[float] = []
    tpot_ms: list[float] = []

    for i in range(steps):
        cold = i == 0
        think = base_ms * scale * (2.5 if cold else 0.8) + (i * 1.5)
        tool = base_ms * scale * 0.4
        decode_tokens = 8 if cold else 4
        planned = think + tool

        t0 = time.perf_counter()
        # Cap sleep for CI friendliness; reported timings must track actual wall.
        time.sleep(min(planned, 50) / 1000.0)
        wall = (time.perf_counter() - t0) * 1000.0

        think_share = (think / planned) if planned > 0 else 0.5
        ttft = wall * think_share
        residual = max(wall - ttft, 0.0)
        tpot = residual / decode_tokens if decode_tokens else residual

        tool_loop_ms.append(wall)
        ttft_ms.append(ttft)
        tpot_ms.extend([tpot] * decode_tokens)

    return {
        "tool_loop_wall_ms": tool_loop_ms,
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "tokens_per_s": None,
        "prefix_cache_hit_rate": None if steps < 2 else round((steps - 1) / steps, 3),
        "vram_peak_mb": None,
        "vram_free_mb": None,
        "tool_loop_p50_ms": statistics.median(tool_loop_ms),
        "ttft_p50_ms": statistics.median(ttft_ms) if ttft_ms else None,
        "tpot_p50_ms": statistics.median(tpot_ms) if tpot_ms else None,
    }


def merge_loops(loops: list[dict]) -> dict:
    """Merge concurrent serial agent loops into one metrics blob."""
    tool_loop: list[float] = []
    ttft: list[float] = []
    tpot: list[float] = []
    for loop in loops:
        tool_loop.extend(loop["tool_loop_wall_ms"])
        ttft.extend(loop["ttft_ms"])
        tpot.extend(loop["tpot_ms"])
    return {
        "tool_loop_wall_ms": tool_loop,
        "ttft_ms": ttft,
        "tpot_ms": tpot,
        "tokens_per_s": None,
        "prefix_cache_hit_rate": loops[0]["prefix_cache_hit_rate"] if loops else None,
        "vram_peak_mb": None,
        "vram_free_mb": None,
        "tool_loop_p50_ms": statistics.median(tool_loop) if tool_loop else None,
        "ttft_p50_ms": statistics.median(ttft) if ttft else None,
        "tpot_p50_ms": statistics.median(tpot) if tpot else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results"),
        help="Output directory for JSON (default: results/)",
    )
    parser.add_argument(
        "--profile",
        default="synthetic_react",
        choices=["synthetic_react", "synthetic_plan_exec", "coding_tool"],
        help="Workload profile name (changes synthetic cost shape)",
    )
    parser.add_argument("--steps", type=int, default=5, help="Agent loop steps (>= 1)")
    parser.add_argument(
        "--base-ms",
        type=float,
        default=5.0,
        help="Base synthetic latency scale (ms)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of sequential agent sessions to simulate (>= 1)",
    )
    args = parser.parse_args()

    if args.steps < 1:
        parser.error("--steps must be >= 1")
    if args.concurrency < 1:
        parser.error("--concurrency must be >= 1")
    if args.base_ms < 0:
        parser.error("--base-ms must be >= 0")

    args.out.mkdir(parents=True, exist_ok=True)
    rid = run_id(args.profile)

    # Concurrency is sequential multi-session simulation (not true parallelism).
    loops = [
        synthetic_loop(args.steps, args.base_ms, args.profile)
        for _ in range(args.concurrency)
    ]
    metrics = merge_loops(loops)

    payload = {
        "schema_version": 1,
        "run_id": rid,
        "timestamp_utc": utc_now(),
        "host": probe_gpu(),
        "engine": {
            "name": "synthetic",
            "version": "harness-0.2",
            "endpoint": None,
        },
        "model": {
            "id": None,
            "quant": None,
            "context_length": None,
        },
        "workload": {
            "profile": args.profile,
            "concurrency": args.concurrency,
            "concurrency_mode": "sequential_sessions",
            "steps": args.steps,
        },
        "metrics": {
            "tool_loop_wall_ms": metrics["tool_loop_wall_ms"],
            "ttft_ms": metrics["ttft_ms"],
            "tpot_ms": metrics["tpot_ms"],
            "tokens_per_s": metrics["tokens_per_s"],
            "prefix_cache_hit_rate": metrics["prefix_cache_hit_rate"],
            "vram_peak_mb": metrics["vram_peak_mb"],
            "vram_free_mb": metrics["vram_free_mb"],
            "tool_loop_p50_ms": metrics["tool_loop_p50_ms"],
            "ttft_p50_ms": metrics["ttft_p50_ms"],
            "tpot_p50_ms": metrics["tpot_p50_ms"],
        },
        "slo": {"met": None, "notes": "Synthetic run; SLOs not applied"},
        "notes": (
            f"Synthetic agent loop without live LLM; "
            f"profile={args.profile} concurrency={args.concurrency} sequential"
        ),
    }

    out_path = args.out / f"{rid}.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")
    print(
        f"tool_loop_p50_ms={metrics['tool_loop_p50_ms']:.2f} "
        f"ttft_p50_ms={metrics['ttft_p50_ms']:.2f} "
        f"tpot_p50_ms={metrics['tpot_p50_ms']:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
