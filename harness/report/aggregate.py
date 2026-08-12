#!/usr/bin/env python3
"""Aggregate results/*.json into a CSV summary."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


COLUMNS = [
    "run_id",
    "timestamp_utc",
    "profile",
    "engine",
    "model_id",
    "concurrency",
    "tool_loop_p50_ms",
    "ttft_p50_ms",
    "tpot_p50_ms",
    "vram_peak_mb",
    "notes",
]

SUPPORTED_SCHEMA = 1


def row_from(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("top-level JSON must be an object")
    schema = data.get("schema_version")
    if schema != SUPPORTED_SCHEMA:
        raise ValueError(f"unsupported schema_version={schema!r} (want {SUPPORTED_SCHEMA})")
    if "metrics" not in data:
        raise ValueError("missing metrics object")
    metrics = data["metrics"]
    if not isinstance(metrics, dict):
        raise ValueError("metrics must be an object")
    workload = data.get("workload") if isinstance(data.get("workload"), dict) else {}
    engine = data.get("engine") if isinstance(data.get("engine"), dict) else {}
    model = data.get("model") if isinstance(data.get("model"), dict) else {}
    return {
        "run_id": data.get("run_id"),
        "timestamp_utc": data.get("timestamp_utc"),
        "profile": workload.get("profile"),
        "engine": engine.get("name"),
        "model_id": model.get("id"),
        "concurrency": workload.get("concurrency"),
        "tool_loop_p50_ms": metrics.get("tool_loop_p50_ms"),
        "ttft_p50_ms": metrics.get("ttft_p50_ms"),
        "tpot_p50_ms": metrics.get("tpot_p50_ms"),
        "vram_peak_mb": metrics.get("vram_peak_mb"),
        "notes": data.get("notes"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="CSV path (default: results/summary.csv)",
    )
    args = parser.parse_args()
    out = args.out or (args.results / "summary.csv")

    rows: list[dict[str, Any]] = []
    for path in sorted(args.results.glob("*.json")):
        try:
            rows.append(row_from(path))
        except (json.JSONDecodeError, OSError, ValueError, TypeError, AttributeError) as exc:
            print(f"skip {path}: {exc}")

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
