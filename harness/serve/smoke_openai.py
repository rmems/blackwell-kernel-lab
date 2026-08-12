#!/usr/bin/env python3
"""OpenAI-compatible smoke + timing for kernel/engine campaigns (#12, #16, #10).

Talks to Ollama or llama-server. Does not implement CUDA kernels — it measures them.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_id(label: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in label)[:40]
    return f"{stamp}-{safe}"


def probe_vram() -> dict:
    out = {"vram_used_mb": None, "vram_free_mb": None, "vram_total_mb": None}
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        ).strip()
        used, free, total = [float(x.strip()) for x in raw.split(",")]
        out["vram_used_mb"] = int(used)
        out["vram_free_mb"] = int(free)
        out["vram_total_mb"] = int(total)
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        pass
    return out


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
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }
    data = json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    t0 = time.perf_counter()
    ttft_ms = None
    content_parts: list[str] = []
    completion_tokens = None

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            if not stream:
                payload = json.loads(resp.read().decode("utf-8"))
                wall_ms = (time.perf_counter() - t0) * 1000.0
                msg = (payload.get("choices") or [{}])[0].get("message") or {}
                text = msg.get("content") or ""
                usage = payload.get("usage") or {}
                return {
                    "ok": True,
                    "content": text,
                    "wall_ms": wall_ms,
                    "ttft_ms": wall_ms,  # non-stream: no separate TTFT
                    "completion_tokens": usage.get("completion_tokens"),
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "raw_error": None,
                }

            # SSE stream
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
                if first:
                    ttft_ms = (time.perf_counter() - t0) * 1000.0
                    first = False
                delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
                piece = delta.get("content")
                if piece:
                    content_parts.append(piece)
            wall_ms = (time.perf_counter() - t0) * 1000.0
            text = "".join(content_parts)
            return {
                "ok": True,
                "content": text,
                "wall_ms": wall_ms,
                "ttft_ms": ttft_ms if ttft_ms is not None else wall_ms,
                "completion_tokens": completion_tokens,
                "prompt_tokens": None,
                "raw_error": None,
            }
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")[:500]
        return {
            "ok": False,
            "content": "",
            "wall_ms": (time.perf_counter() - t0) * 1000.0,
            "ttft_ms": None,
            "completion_tokens": None,
            "prompt_tokens": None,
            "raw_error": f"HTTP {e.code}: {err}",
        }
    except urllib.error.URLError as e:
        return {
            "ok": False,
            "content": "",
            "wall_ms": (time.perf_counter() - t0) * 1000.0,
            "ttft_ms": None,
            "completion_tokens": None,
            "prompt_tokens": None,
            "raw_error": str(e.reason),
        }


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
    args = p.parse_args()

    prompt = args.prompt
    if args.agent_prompt:
        prompt = (
            "You are a local coding agent. Tools: read_file, run_shell. "
            "User: confirm you can call tools by replying with exactly: kernel-smoke-ok"
        )

    vram_before = probe_vram()
    result = chat_completion(
        args.base_url,
        args.model,
        args.api_key,
        prompt,
        args.max_tokens,
        0.0,
        args.stream,
    )
    vram_after = probe_vram()

    rid = run_id(args.label)
    payload = {
        "schema_version": 1,
        "run_id": rid,
        "timestamp_utc": utc_now(),
        "host": {
            "gpu_name": "NVIDIA GeForce RTX 5080",
            "vram_total_mb": vram_after.get("vram_total_mb") or vram_before.get("vram_total_mb"),
            "driver": None,
            "cuda": None,
        },
        "engine": {
            "name": "openai-compat",
            "version": None,
            "endpoint": args.base_url,
            "label": args.label,
        },
        "model": {
            "id": args.model,
            "quant": None,
            "context_length": None,
        },
        "workload": {
            "profile": "engine_smoke" if not args.agent_prompt else "agent_shaped_smoke",
            "concurrency": 1,
            "steps": 1,
            "stream": args.stream,
        },
        "metrics": {
            "tool_loop_wall_ms": [result["wall_ms"]],
            "ttft_ms": [result["ttft_ms"]] if result["ttft_ms"] is not None else [],
            "tpot_ms": [],
            "tokens_per_s": None,
            "prefix_cache_hit_rate": None,
            "vram_peak_mb": vram_after.get("vram_used_mb"),
            "vram_free_mb": vram_after.get("vram_free_mb"),
            "vram_before_mb": vram_before.get("vram_used_mb"),
            "wall_ms": result["wall_ms"],
            "ttft_ms_scalar": result["ttft_ms"],
            "completion_tokens": result["completion_tokens"],
            "prompt_tokens": result["prompt_tokens"],
        },
        "slo": {
            "met": result["ok"] and "kernel-smoke-ok" in (result["content"] or ""),
            "notes": "smoke content check" if result["ok"] else result["raw_error"],
        },
        "notes": (
            "ok={} content={!r} stream={} label={}".format(
                result["ok"],
                (result["content"] or "")[:120],
                args.stream,
                args.label,
            )
        ),
        "kernel_campaign": {
            "layer": "L1",
            "issue": "#12/#16",
            "meaning": "Measure engine CUDA paths; not custom .cu",
        },
    }

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{rid}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}")
    print(
        f"ok={result['ok']} wall_ms={result['wall_ms']:.1f} "
        f"ttft_ms={result['ttft_ms']} "
        f"vram_used={vram_after.get('vram_used_mb')} free={vram_after.get('vram_free_mb')}"
    )
    if result["raw_error"]:
        print(f"error: {result['raw_error']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
