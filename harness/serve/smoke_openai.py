#!/usr/bin/env python3
"""OpenAI-compatible smoke + timing for kernel/engine campaigns (#12, #16, #10).

Talks to Ollama or llama-server. Does not implement CUDA kernels — it measures them.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
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


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_id(label: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in label)[:40]
    return f"{stamp}-{safe}-{uuid.uuid4().hex[:8]}"


def probe_gpu() -> dict:
    """Probe name + memory via nvidia-smi (portable across hosts)."""
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


def _err_result(t0: float, msg: str) -> dict:
    return {
        "ok": False,
        "content": "",
        "wall_ms": (time.perf_counter() - t0) * 1000.0,
        "ttft_ms": None,
        "completion_tokens": None,
        "prompt_tokens": None,
        "raw_error": msg,
    }


def chat_completion(
    base_url: str,
    model: str,
    api_key: str | None,
    prompt: str,
    max_tokens: int,
    temperature: float,
    stream: bool,
) -> dict:
    url = base_url.rstrip("/") + "/chat/completions"
    body: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }
    if stream:
        # OpenAI-compatible servers that support it may return usage on the final chunk.
        body["stream_options"] = {"include_usage": True}

    data = json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    t0 = time.perf_counter()
    ttft_ms = None
    content_parts: list[str] = []
    completion_tokens = None
    prompt_tokens = None

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            if not stream:
                raw = resp.read().decode("utf-8")
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError as e:
                    return _err_result(t0, f"JSONDecodeError: {e}")
                if not isinstance(payload, dict):
                    return _err_result(t0, "response is not a JSON object")
                wall_ms = (time.perf_counter() - t0) * 1000.0
                choices = payload.get("choices")
                if not isinstance(choices, list) or not choices:
                    return _err_result(t0, "missing choices[]")
                msg = choices[0].get("message") if isinstance(choices[0], dict) else {}
                if not isinstance(msg, dict):
                    msg = {}
                content = msg.get("content")
                if content is None:
                    text = ""
                elif isinstance(content, str):
                    text = content
                else:
                    return _err_result(t0, f"message.content is {type(content).__name__}, expected str")
                usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
                # Non-stream: no first-token timestamp — leave ttft_ms null so TPOT
                # is not faked as 0.0 (wall - wall).
                return {
                    "ok": True,
                    "content": text,
                    "wall_ms": wall_ms,
                    "ttft_ms": None,
                    "completion_tokens": usage.get("completion_tokens"),
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "raw_error": None,
                }

            first = True
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data_s = line[5:].strip()
                if data_s == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_s)
                except json.JSONDecodeError:
                    continue
                if not isinstance(chunk, dict):
                    continue
                if first:
                    ttft_ms = (time.perf_counter() - t0) * 1000.0
                    first = False
                usage = chunk.get("usage")
                if isinstance(usage, dict):
                    completion_tokens = usage.get("completion_tokens", completion_tokens)
                    prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                choices = chunk.get("choices")
                if not isinstance(choices, list) or not choices:
                    continue
                delta = choices[0].get("delta") if isinstance(choices[0], dict) else {}
                if not isinstance(delta, dict):
                    continue
                piece = delta.get("content")
                if piece is None:
                    continue
                if not isinstance(piece, str):
                    return _err_result(
                        t0, f"delta.content is {type(piece).__name__}, expected str"
                    )
                content_parts.append(piece)
            wall_ms = (time.perf_counter() - t0) * 1000.0
            text = "".join(content_parts)
            return {
                "ok": True,
                "content": text,
                "wall_ms": wall_ms,
                "ttft_ms": ttft_ms if ttft_ms is not None else wall_ms,
                "completion_tokens": completion_tokens,
                "prompt_tokens": prompt_tokens,
                "raw_error": None,
            }
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")[:500]
        return _err_result(t0, f"HTTP {e.code}: {err}")
    except urllib.error.URLError as e:
        return _err_result(t0, str(e.reason))
    except (TimeoutError, socket.timeout) as e:
        return _err_result(t0, f"timeout: {e}")
    except (json.JSONDecodeError, KeyError, IndexError, TypeError, AttributeError) as e:
        return _err_result(t0, f"{type(e).__name__}: {e}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--base-url", default=os.environ.get("BKL_BASE_URL", "http://127.0.0.1:11434/v1"))
    p.add_argument("--model", default=os.environ.get("BKL_MODEL", "granite4.1:8b"))
    p.add_argument("--api-key", default=os.environ.get("BKL_API_KEY") or None)
    p.add_argument("--label", default="engine-smoke")
    p.add_argument("--stream", action="store_true", help="SSE stream for TTFT")
    p.add_argument("--max-tokens", type=int, default=int(os.environ.get("BKL_MAX_TOKENS", "64")))
    p.add_argument(
        "--prompt",
        default="Reply with exactly: kernel-smoke-ok",
    )
    p.add_argument(
        "--agent-prompt",
        action="store_true",
        help="Use a short agent-shaped prompt (system-like tools blurb + user)",
    )
    p.add_argument(
        "--profile",
        default=None,
        choices=profiles.names(),
        help=(
            "Use a named workload profile's cold-prefill turn as the prompt "
            "(docs/WORKLOADS.md). Single turn only — no resume phase. "
            "Takes precedence over --agent-prompt/--prompt."
        ),
    )
    p.add_argument(
        "--engine-version",
        default=os.environ.get("BKL_ENGINE_VERSION"),
        help="Optional engine binary/version string for ablation provenance",
    )
    p.add_argument(
        "--engine-flags",
        default=os.environ.get("BKL_ENGINE_FLAGS"),
        help="Optional flags string for ablation provenance",
    )
    p.add_argument(
        "--quant",
        default=os.environ.get("BKL_QUANT"),
        help="Optional quant label for ablation provenance",
    )
    p.add_argument(
        "--min-free-vram-mb",
        type=int,
        default=int(os.environ.get("BKL_MIN_FREE_VRAM_MB", "0")),
        help="If >0, fail when post-load free VRAM is below this (e.g. 2048)",
    )
    p.add_argument(
        "--skip-content-slo",
        action="store_true",
        help="Record metrics even when reply is not exactly kernel-smoke-ok (ablations)",
    )
    args = p.parse_args()

    prompt = args.prompt
    if args.agent_prompt:
        prompt = (
            "You are a local coding agent. Tools: read_file, run_shell. "
            "User: confirm you can call tools by replying with exactly: kernel-smoke-ok"
        )
    if args.profile:
        # Same cold-prefill text the live harness (#10) sends, so an ablation cell
        # and a live profile row measure the same prompt.
        prompt = profiles.cold_prompt(args.profile)
        if not args.skip_content_slo:
            p.error("--profile prompts do not emit kernel-smoke-ok; pass --skip-content-slo")

    if args.profile:
        workload_profile = f"{args.profile}_cold_only"
    elif args.agent_prompt:
        workload_profile = "agent_shaped_smoke"
    else:
        workload_profile = "engine_smoke"

    vram_before = probe_gpu()
    result = chat_completion(
        args.base_url,
        args.model,
        args.api_key,
        prompt,
        args.max_tokens,
        0.0,
        args.stream,
    )
    vram_after = probe_gpu()

    used_vals = [
        v
        for v in (vram_before.get("vram_used_mb"), vram_after.get("vram_used_mb"))
        if isinstance(v, int)
    ]
    # Honest label: max of samples, not a continuous peak sampler.
    vram_sample_max = max(used_vals) if used_vals else None

    # Exact smoke token (strip); substring would accept "cannot return kernel-smoke-ok".
    content_ok = (result["content"] or "").strip() == "kernel-smoke-ok"
    if args.skip_content_slo:
        slo_met = bool(result["ok"])
    else:
        slo_met = bool(result["ok"] and content_ok)
    free_after = vram_after.get("vram_free_mb")
    if args.min_free_vram_mb > 0:
        if not isinstance(free_after, int):
            slo_met = False
            result["raw_error"] = (
                (result["raw_error"] or "") + " VRAM probe unavailable; cannot enforce min free"
            ).strip()
        elif free_after < args.min_free_vram_mb:
            slo_met = False
            result["raw_error"] = (
                (result["raw_error"] or "")
                + f" free VRAM {free_after} MiB < min {args.min_free_vram_mb}"
            ).strip()

    # Derive TPOT only when we have a real first-token timestamp (streaming).
    # Non-stream leaves ttft_ms null; never emit a misleading 0.0 TPOT.
    tpot_ms_val = None
    ct = result.get("completion_tokens")
    ttft_s = result.get("ttft_ms")
    wall_s = result.get("wall_ms")
    if (
        args.stream
        and isinstance(ct, int)
        and ct > 0
        and isinstance(ttft_s, (int, float))
        and isinstance(wall_s, (int, float))
        and float(wall_s) > float(ttft_s)
    ):
        decode_ms = float(wall_s) - float(ttft_s)
        denom = max(ct - 1, 1)
        tpot_ms_val = decode_ms / denom

    rid = run_id(args.label)
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
            "name": "openai-compat",
            "version": args.engine_version,
            "endpoint": args.base_url,
            "label": args.label,
            "flags": args.engine_flags,
        },
        "model": {
            "id": args.model,
            "quant": args.quant,
            "context_length": None,
        },
        "workload": {
            "profile": workload_profile,
            "concurrency": 1,
            "steps": 1,
            "stream": args.stream,
            "phases": ["cold_prefill", "short_decode"],
        },
        "metrics": {
            "tool_loop_wall_ms": [result["wall_ms"]],
            "ttft_ms": [result["ttft_ms"]] if result["ttft_ms"] is not None else [],
            "tpot_ms": [tpot_ms_val] if tpot_ms_val is not None else [],
            # aggregate.py CSV column; single-sample smoke → p50 equals the sample.
            "tpot_p50_ms": tpot_ms_val,
            "tokens_per_s": (1000.0 / tpot_ms_val) if tpot_ms_val and tpot_ms_val > 0 else None,
            "prefix_cache_hit_rate": None,
            # Max of before/after samples — not continuous peak sampling.
            "vram_peak_mb": vram_sample_max,
            "vram_sample_note": "max(before, after) samples; not continuous peak",
            "vram_free_mb": free_after,
            "vram_before_mb": vram_before.get("vram_used_mb"),
            "vram_after_mb": vram_after.get("vram_used_mb"),
            "wall_ms": result["wall_ms"],
            "ttft_ms_scalar": result["ttft_ms"],
            "tpot_ms_scalar": tpot_ms_val,
            "completion_tokens": result["completion_tokens"],
            "prompt_tokens": result["prompt_tokens"],
        },
        "slo": {
            "met": slo_met,
            "notes": (
                "request ok (content SLO skipped)"
                if args.skip_content_slo and result["ok"]
                else (
                    "smoke content check"
                    if result["ok"] and content_ok
                    else (result["raw_error"] or "content mismatch / request failed")
                )
            ),
        },
        "notes": (
            "ok={} content={!r} stream={} label={} flags={!r} quant={!r}".format(
                result["ok"],
                (result["content"] or "")[:120],
                args.stream,
                args.label,
                args.engine_flags,
                args.quant,
            )
        ),
        "kernel_campaign": {
            "layer": "L1",
            "issue": "#16",
            "meaning": "Measure engine CUDA paths; not custom .cu",
            "engine_version": args.engine_version,
            "engine_flags": args.engine_flags,
            "quant": args.quant,
        },
    }

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{rid}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}")
    print(
        f"ok={result['ok']} slo_met={slo_met} wall_ms={result['wall_ms']:.1f} "
        f"ttft_ms={result['ttft_ms']} "
        f"vram_used_after={vram_after.get('vram_used_mb')} free={free_after}"
    )
    if result["raw_error"]:
        print(f"error: {result['raw_error']}")
    if not slo_met:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
