"""Minimal OpenAI-compatible chat client for lab harnesses (#3, #10).

Streaming path records TTFT at the **first non-empty content token**.
No third-party deps.
"""

from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
from typing import Any


def chat_completion(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    *,
    api_key: str | None = None,
    max_tokens: int = 64,
    temperature: float = 0.0,
    stream: bool = True,
    timeout: float = 300.0,
) -> dict[str, Any]:
    """POST /chat/completions. Returns ok, content, wall_ms, ttft_ms, usage, raw_error."""
    url = base_url.rstrip("/") + "/chat/completions"
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }
    if stream:
        body["stream_options"] = {"include_usage": True}

    data = json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    t0 = time.perf_counter()
    ttft_ms: float | None = None
    content_parts: list[str] = []
    completion_tokens = None
    prompt_tokens = None

    def _err(msg: str) -> dict[str, Any]:
        return {
            "ok": False,
            "content": "",
            "wall_ms": (time.perf_counter() - t0) * 1000.0,
            "ttft_ms": None,
            "completion_tokens": None,
            "prompt_tokens": None,
            "raw_error": msg,
        }

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if not stream:
                raw = resp.read().decode("utf-8")
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError as e:
                    return _err(f"JSONDecodeError: {e}")
                if not isinstance(payload, dict):
                    return _err("response is not a JSON object")
                wall_ms = (time.perf_counter() - t0) * 1000.0
                choices = payload.get("choices")
                if not isinstance(choices, list) or not choices:
                    return _err("missing choices[]")
                msg = choices[0].get("message") if isinstance(choices[0], dict) else {}
                if not isinstance(msg, dict):
                    msg = {}
                content = msg.get("content")
                text = content if isinstance(content, str) else ("" if content is None else None)
                if text is None:
                    return _err(f"message.content is {type(content).__name__}, expected str")
                usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
                return {
                    "ok": True,
                    "content": text,
                    "wall_ms": wall_ms,
                    "ttft_ms": None,
                    "completion_tokens": usage.get("completion_tokens"),
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "raw_error": None,
                }

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
                    return _err(f"delta.content is {type(piece).__name__}, expected str")
                # TTFT = first non-empty content token (ignore role-only chunks).
                if ttft_ms is None and piece:
                    ttft_ms = (time.perf_counter() - t0) * 1000.0
                content_parts.append(piece)
            wall_ms = (time.perf_counter() - t0) * 1000.0
            text = "".join(content_parts)
            if not text and ttft_ms is None:
                return _err("stream ended without content tokens")
            return {
                "ok": True,
                "content": text,
                "wall_ms": wall_ms,
                "ttft_ms": ttft_ms,
                "completion_tokens": completion_tokens,
                "prompt_tokens": prompt_tokens,
                "raw_error": None,
            }
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")[:500]
        return _err(f"HTTP {e.code}: {err}")
    except urllib.error.URLError as e:
        return _err(str(e.reason))
    except (TimeoutError, socket.timeout) as e:
        return _err(f"timeout: {e}")
    except (json.JSONDecodeError, KeyError, IndexError, TypeError, AttributeError) as e:
        return _err(f"{type(e).__name__}: {e}")


def derive_tpot_ms(
    wall_ms: float | None, ttft_ms: float | None, completion_tokens: int | None
) -> float | None:
    if (
        not isinstance(completion_tokens, int)
        or completion_tokens <= 0
        or not isinstance(ttft_ms, (int, float))
        or not isinstance(wall_ms, (int, float))
    ):
        return None
    decode_ms = max(float(wall_ms) - float(ttft_ms), 0.0)
    return decode_ms / max(completion_tokens - 1, 1)


def infer_engine_name(base_url: str, explicit: str | None) -> str:
    """Prefer explicit BKL_ENGINE / --engine-name; else guess from URL."""
    if explicit and explicit not in ("", "openai-compat"):
        return explicit
    u = (base_url or "").lower()
    if "11434" in u or "ollama" in u:
        return "ollama"
    if "8080" in u or "llama" in u:
        return "llama-server"
    return explicit or "openai-compat"
