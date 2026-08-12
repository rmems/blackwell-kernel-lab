#!/usr/bin/env python3
"""Aggregate results/*.json into a CSV summary."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


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


def row_from(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    metrics = data.get("metrics") or {}
    return {
        "run_id": data.get("run_id"),
        "timestamp_utc": data.get("timestamp_utc"),
        "profile": (data.get("workload") or {}).get("profile"),
        "engine": (data.get("engine") or {}).get("name"),
        "model_id": (data.get("model") or {}).get("id"),
        "concurrency": (data.get("workload") or {}).get("concurrency"),
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

    rows = []
    for path in sorted(args.results.glob("*.json")):
        try:
            rows.append(row_from(path))
        except (json.JSONDecodeError, OSError) as exc:
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
