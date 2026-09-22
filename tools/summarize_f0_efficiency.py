#!/usr/bin/env python3
"""Derive F0 training-efficiency and GPU-headroom summaries (CPU, no GPU).

Reads a #52/#53 correlation bundle (Agoge markers + BKL GPU samples) and
emits `bkl.f0_efficiency.v1` JSON plus a concise text report. Formulas:
docs/f0-efficiency-metrics.md.

Usage:
    python3 tools/summarize_f0_efficiency.py \\
      --markers fixtures/f0-correlation/agoge-markers.jsonl \\
      --samples fixtures/f0-correlation/bkl-gpu-samples.jsonl \\
      --out results/f0-efficiency.json \\
      --report results/f0-efficiency.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import check_f0_correlation as f0corr
from f0_efficiency_compare import (
    load_optional_json,
    load_optional_jsonl,
    partition_physical_samples,
    profile_window_summaries,
    resolve_run_id,
    run_identity,
    selected_host_gpu,
    train_fit_comparison,
    unjoined_samples,
    validate_bundle,
)
from f0_efficiency_numbers import (
    ENERGY_NOTE,
    HEADROOM_FLOOR_MIB,
    MAX_GAP_FACTOR,
    SCHEMA_OUT,
    SummaryError,
    coverage_block,
    distribution,
    energy_rates,
    expected_sample_count,
    field_block,
    field_inventory,
    filter_run,
    global_step_delta_sum,
    integrate_board_power,
    ok_value,
    phase_metrics,
    require_max_gap_factor,
    samples_by_phase,
    step_durations_s,
    throughput_from_markers,
    vram_block,
)
from f0_efficiency_report import write_json, write_report

__all__ = [
    "ENERGY_NOTE",
    "HEADROOM_FLOOR_MIB",
    "MAX_GAP_FACTOR",
    "SCHEMA_OUT",
    "SummaryError",
    "energy_rates",
    "expected_sample_count",
    "integrate_board_power",
    "main",
    "ok_value",
    "parse_args",
    "summarize_paths",
    "summarize_run",
    "throughput_from_markers",
]


def physical_field_blocks(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "vram": vram_block(samples),
        "power": field_block(samples, "power", avg_key="avg_w", peak_key="peak_w", min_key=None, unit="W"),
        "temperature_gpu": field_block(
            samples, "temperature_gpu", avg_key="avg_c", peak_key="peak_c", min_key=None, unit="C"
        ),
        "utilization_gpu": field_block(
            samples, "utilization_gpu", avg_key="avg", peak_key="peak", min_key=None, unit="%"
        ),
        "utilization_memory": field_block(
            samples, "utilization_memory", avg_key="avg", peak_key="peak", min_key=None, unit="%"
        ),
    }


def summarize_run(
    markers: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    *,
    max_gap_factor: float,
    profile_rows: list[dict[str, Any]],
    train_fit_ref: dict[str, Any] | None,
) -> dict[str, Any]:
    host, gpu = selected_host_gpu(markers, samples)
    physical, excluded = partition_physical_samples(samples, host, gpu)
    pairs = f0corr.join_samples(markers, physical)
    throughput = throughput_from_markers(markers)
    durations = step_durations_s(markers)
    energy = energy_rates(
        integrate_board_power(physical, max_gap_factor),
        throughput,
        global_step_delta_sum(markers),
        markers=markers,
        samples=physical,
    )
    fields = physical_field_blocks(physical)
    run_id = markers[0]["agoge_run_id"] if markers else physical[0]["agoge_run_id"]
    return {
        "schema_version": SCHEMA_OUT,
        "energy_is_approximate": True,
        "hardware_note": (
            "RTX 5080 / sm_120 / ~16 GB. Leave ≥2 GiB free. "
            "Derived measurements, not a trainer or safety controller."
        ),
        "run": run_identity(markers, physical or samples, run_id),
        "coverage": coverage_block(markers, physical, pairs, energy["long_gaps"]),
        "throughput": throughput,
        "step_time": distribution(durations, "s"),
        **fields,
        "energy": energy,
        "by_phase": {name: phase_metrics(group) for name, group in samples_by_phase(pairs).items()},
        "unjoined_sample_count": len(unjoined_samples(physical, pairs)),
        "excluded_foreign_sample_count": excluded,
        "profile_windows": profile_window_summaries(physical, pairs, profile_rows, run_id),
        "field_status_counts": field_inventory(physical),
        "train_fit_comparison": train_fit_comparison(fields["vram"], train_fit_ref),
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markers", type=Path, default=Path("fixtures/f0-correlation/agoge-markers.jsonl"))
    parser.add_argument("--samples", type=Path, default=Path("fixtures/f0-correlation/bkl-gpu-samples.jsonl"))
    parser.add_argument("--profile-windows", type=Path, default=None)
    parser.add_argument("--train-fit-ref", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--out", type=Path, default=None, help="Write JSON summary here")
    parser.add_argument("--report", type=Path, default=None, help="Write markdown report here")
    parser.add_argument("--max-gap-factor", type=float, default=MAX_GAP_FACTOR)
    parser.add_argument("--print-json", action="store_true", help="Print JSON to stdout")
    return parser.parse_args(argv[1:])


def summarize_paths(
    markers_path: Path,
    samples_path: Path,
    *,
    run_id: str | None = None,
    max_gap_factor: float = MAX_GAP_FACTOR,
    profile_windows_path: Path | None = None,
    train_fit_ref_path: Path | None = None,
) -> dict[str, Any]:
    require_max_gap_factor(max_gap_factor)
    markers = f0corr.load_jsonl(markers_path)
    samples = f0corr.load_jsonl(samples_path)
    validate_bundle(markers, samples)
    chosen = resolve_run_id(markers, samples, run_id)
    markers = filter_run(markers, chosen)
    samples = filter_run(samples, chosen)
    return summarize_run(
        markers,
        samples,
        max_gap_factor=max_gap_factor,
        profile_rows=load_optional_jsonl(profile_windows_path),
        train_fit_ref=load_optional_json(train_fit_ref_path),
    )


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        summary = summarize_paths(
            args.markers,
            args.samples,
            run_id=args.run_id,
            max_gap_factor=args.max_gap_factor,
            profile_windows_path=args.profile_windows,
            train_fit_ref_path=args.train_fit_ref,
        )
    except (f0corr.SchemaError, SummaryError, OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(error, file=sys.stderr)
        return 1
    json_text = write_json(args.out, summary)
    report_text = write_report(args.report, summary)
    if args.print_json:
        sys.stdout.write(json_text)
    else:
        sys.stdout.write(report_text)
    cov = summary["coverage"]
    print(
        f"{args.markers}: {cov['marker_count']} markers; "
        f"{args.samples}: {cov['sample_count']} samples; "
        f"joined={cov['joined_count']} coverage_ratio={cov['coverage_ratio']}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
