#!/usr/bin/env python3
"""Fixture and formula checks for F0 efficiency summaries (CPU, no GPU).

See docs/f0-efficiency-metrics.md. Used by .github/workflows/ci-cpu.yml.

Usage:
    python3 tools/check_f0_efficiency.py
"""

from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path
from typing import Any

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import check_f0_correlation as f0corr
import summarize_f0_efficiency as f0eff

ROOT = Path(__file__).resolve().parents[1]
MARKERS = ROOT / "fixtures/f0-correlation/agoge-markers.jsonl"
SAMPLES = ROOT / "fixtures/f0-correlation/bkl-gpu-samples.jsonl"
PROFILE = ROOT / "fixtures/f0-efficiency/profile-windows.jsonl"
TRAIN_FIT = ROOT / "fixtures/f0-efficiency/train-fit-reference.json"
EXAMPLE = ROOT / "fixtures/f0-efficiency/minicpm5-summary.json"


class CheckError(Exception):
    """A formula or fixture assertion failed."""


def require(condition: object, message: str) -> None:
    if not condition:
        raise CheckError(message)


def close(actual: float | None, expected: float, message: str) -> None:
    require(actual is not None, f"{message}: got None, expected {expected}")
    require(math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-9), f"{message}: {actual} != {expected}")


def load_minicpm5() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    markers = f0corr.load_jsonl(MARKERS)
    samples = f0corr.load_jsonl(SAMPLES)
    for rec in markers:
        f0corr.check_marker(rec)
    for rec in samples:
        f0corr.check_sample(rec)
    return markers, samples


def summarize_minicpm5() -> dict[str, Any]:
    return f0eff.summarize_paths(
        MARKERS,
        SAMPLES,
        profile_windows_path=PROFILE,
        train_fit_ref_path=TRAIN_FIT,
    )


def clone(record: dict[str, Any], **updates: Any) -> dict[str, Any]:
    out = copy.deepcopy(record)
    out.update(updates)
    return out


def set_measurement(record: dict[str, Any], field: str, value: float | None, unit: str, status: str) -> None:
    record[field] = {"value": value, "unit": unit, "status": status}


def check_minicpm5_machine_readable() -> None:
    summary = summarize_minicpm5()
    require(summary["schema_version"] == f0eff.SCHEMA_OUT, "unexpected summary schema")
    require(summary["run"]["agoge_run_id"] == "run_minicpm5_fixture_001", "unexpected run id")
    require(summary["run"]["formulas_are_model_agnostic"] is True, "formulas must be model-agnostic")
    require(EXAMPLE.is_file(), f"missing example JSON {EXAMPLE}")
    example = json.loads(EXAMPLE.read_text())
    require(example["run"]["agoge_run_id"] == summary["run"]["agoge_run_id"], "example run id drifted")
    require(example["vram"]["peak_used_mib"] == summary["vram"]["peak_used_mib"], "example peak VRAM drifted")


def ok_field_values(samples: list[dict[str, Any]], field: str) -> list[float]:
    return [value for value in (f0eff.ok_value(sample[field]) for sample in samples) if value is not None]


def check_peak_vram_matches_samples() -> None:
    markers, samples = load_minicpm5()
    summary = f0eff.summarize_run(markers, samples, max_gap_factor=3.0, profile_rows=[], train_fit_ref=None)
    close(summary["vram"]["peak_used_mib"], max(ok_field_values(samples, "vram_used")), "peak VRAM")
    close(summary["vram"]["min_free_mib"], min(ok_field_values(samples, "vram_free")), "min free")
    close(summary["vram"]["min_headroom_mib"], min(ok_field_values(samples, "headroom")), "min headroom")
    require(summary["vram"]["peak_used_mib"] == 9800, "fixture peak used is 9800 MiB")
    require(summary["vram"]["min_headroom_mib"] == 6503, "fixture min headroom is 6503 MiB")


def check_tokens_from_agoge_not_gpu() -> None:
    markers, samples = load_minicpm5()
    baseline = f0eff.summarize_run(markers, samples, max_gap_factor=3.0, profile_rows=[], train_fit_ref=None)
    close(baseline["throughput"]["tokens_per_s"], 64.0, "tokens/s from 128 tokens / 2.0 s")
    close(baseline["throughput"]["examples_per_s"], 0.5, "examples/s from 1 example / 2.0 s")
    close(baseline["step_time"]["mean"], 2.0, "one Agoge step interval")
    require(baseline["throughput"]["source"] == "agoge_counters_and_monotonic_ns", "wrong throughput source")
    require(baseline["throughput"]["not_inferred_from_gpu_load"] is True, "must not use GPU load")
    hot = [clone(sample) for sample in samples]
    for sample in hot:
        set_measurement(sample, "utilization_gpu", 100.0, "%", "ok")
    hot_summary = f0eff.summarize_run(markers, hot, max_gap_factor=3.0, profile_rows=[], train_fit_ref=None)
    close(hot_summary["throughput"]["tokens_per_s"], baseline["throughput"]["tokens_per_s"], "util must not change tokens/s")
    close(hot_summary["step_time"]["mean"], baseline["step_time"]["mean"], "util must not change step-time")


def check_energy_skips_long_gap() -> None:
    summary = summarize_minicpm5()
    energy = summary["energy"]
    require(energy["label"] == "approximate", "energy must be labeled approximate")
    require(energy["method"] == "trapezoidal_board_power", "unexpected integration method")
    require(energy["pairs_integrated"] == 0, "MiniCPM5 fixture pair is a long gap")
    require(energy["pairs_skipped_long_gap"] == 1, "expected one long-gap skip")
    require(energy["approximate_joules"] is None, "missing energy must not become 0 J")
    require(energy["approximate_joules_per_step"] is None, "J/step requires an integral")
    require("APPROXIMATE" in energy["note"], "caveat missing")
    require(summary["coverage"]["coverage_ratio"] is not None, "coverage_ratio required")
    close(summary["coverage"]["coverage_ratio"], 0.5, "2 observed / 4 expected")


def check_energy_trapezoid_regular() -> None:
    markers, samples = load_minicpm5()
    template = samples[0]
    grid = []
    powers = (100.0, 200.0, 100.0)
    for index, power in enumerate(powers):
        sample = clone(template, monotonic_ns=1_500_000_000 + index * 500_000_000)
        set_measurement(sample, "power", power, "W", "ok")
        sample["cadence_ms"] = 500
        grid.append(sample)
    energy = f0eff.integrate_board_power(grid, 3.0)
    close(energy["approximate_joules"], 150.0, "0.5*(100+200)*0.5 + 0.5*(200+100)*0.5")
    require(energy["pairs_integrated"] == 2, "both 500 ms pairs should integrate")
    require(energy["pairs_skipped_long_gap"] == 0, "no long gaps on the 500 ms grid")


def check_missing_power_is_not_zero() -> None:
    _markers, samples = load_minicpm5()
    template = samples[0]
    prev = clone(template, monotonic_ns=1_000_000_000)
    cur = clone(template, monotonic_ns=1_500_000_000)
    set_measurement(prev, "power", 100.0, "W", "ok")
    set_measurement(cur, "power", None, "W", "unavailable")
    energy = f0eff.integrate_board_power([prev, cur], 3.0)
    require(energy["pairs_skipped_missing_power"] == 1, "unavailable power pair must skip")
    require(energy["approximate_joules"] is None, "unavailable power must not integrate as 0 W")


def check_missing_counts() -> None:
    summary = summarize_minicpm5()
    temp_counts = summary["field_status_counts"]["temperature_gpu"]
    mem_counts = summary["field_status_counts"]["temperature_memory"]
    require(temp_counts["ok"] == 1, "one ok GPU temp")
    require(temp_counts["unavailable"] == 1, "one unavailable GPU temp")
    require(mem_counts["unsupported"] == 2, "memory temp unsupported on both samples")
    require(summary["temperature_gpu"]["missing_count"] == 1, "temp missing_count")
    require(summary["coverage"]["unjoined_count"] == 0, "fixture samples join")
    require("missing_sample_policy" in summary["coverage"], "policy must be recorded")


def check_granite_same_formulas() -> None:
    markers, samples = load_minicpm5()
    granite = [clone(marker, model_id="ibm-granite/granite-4.1") for marker in markers]
    for marker in granite:
        f0corr.check_marker(marker)
    mini = f0eff.summarize_run(markers, samples, max_gap_factor=3.0, profile_rows=[], train_fit_ref=None)
    other = f0eff.summarize_run(granite, samples, max_gap_factor=3.0, profile_rows=[], train_fit_ref=None)
    require(other["run"]["model_id"] == "ibm-granite/granite-4.1", "model_id should pass through")
    close(other["throughput"]["tokens_per_s"], mini["throughput"]["tokens_per_s"], "same counters")
    close(other["vram"]["peak_used_mib"], mini["vram"]["peak_used_mib"], "same VRAM extrema")
    close(other["step_time"]["mean"], mini["step_time"]["mean"], "same step times")
    source = Path(f0eff.__file__).read_text()
    require("minicpm" not in source.lower(), "summarizer must not special-case MiniCPM")
    require("granite" not in source.lower(), "summarizer must not special-case Granite")


def check_train_fit_not_overwritten() -> None:
    summary = summarize_minicpm5()
    fit = summary["train_fit_comparison"]
    require(fit["derived_is_evidence_only"] is True, "derived must not become SoT")
    require(fit["comparable_peak"] is False, "trainer allocated vs sampler used")
    require(fit["peak_vram"]["reason"] == "different_measurement_basis", "must flag basis mismatch")
    require(fit["peak_vram"]["agreement"] is None, "do not claim agreement across bases")
    require("#44" in fit["source_of_truth"], "must cite train-fit issue")
    same_basis = {
        "peak_allocated_mib": 9800,
        "peak_basis": "bkl_gpu_sample.vram_used",
        "min_free_mib": 6503,
        "min_free_basis": "bkl_gpu_sample.vram_free",
    }
    markers, samples = load_minicpm5()
    agreed = f0eff.summarize_run(
        markers, samples, max_gap_factor=3.0, profile_rows=[], train_fit_ref=same_basis
    )
    require(agreed["train_fit_comparison"]["comparable_peak"] is True, "same basis is comparable")
    require(agreed["train_fit_comparison"]["peak_vram"]["agreement"] is True, "9800 vs 9800")
    require(agreed["train_fit_comparison"]["min_headroom"]["agreement"] is True, "6503 vs 6503")


def check_phase_and_profile() -> None:
    markers, samples = load_minicpm5()
    eval_marker = clone(markers[-1], phase="eval", global_step=2, monotonic_ns=4_000_000_000)
    eval_marker["timestamp_utc"] = "2026-09-14T18:00:04Z"
    eval_sample = clone(samples[-1], monotonic_ns=4_250_000_000)
    eval_sample["timestamp_utc"] = "2026-09-14T18:00:04.250Z"
    eval_sample["profile_window_ref"] = None
    set_measurement(eval_sample, "utilization_gpu", 11.0, "%", "ok")
    bundle_markers = markers + [eval_marker]
    bundle_samples = samples + [eval_sample]
    for rec in bundle_markers:
        f0corr.check_marker(rec)
    for rec in bundle_samples:
        f0corr.check_sample(rec)
    profile_rows = f0corr.load_jsonl(PROFILE)
    summary = f0eff.summarize_run(
        bundle_markers, bundle_samples, max_gap_factor=3.0, profile_rows=profile_rows, train_fit_ref=None
    )
    require(set(summary["by_phase"]) == {"train", "eval"}, f"phases {summary['by_phase'].keys()}")
    close(summary["by_phase"]["eval"]["utilization_gpu"]["avg"], 11.0, "eval util")
    refs = [window["profile_window_ref"] for window in summary["profile_windows"]]
    require("fwd-window-fixture-1" in refs, "profile ref from samples")
    window = next(item for item in summary["profile_windows"] if item["profile_window_ref"] == "fwd-window-fixture-1")
    require(window["compact_summary"]["kernel_count"] == 4, "compact #54 summary should join")


def check_cli_minicpm5() -> None:
    argv = [
        "summarize_f0_efficiency.py",
        "--markers",
        str(MARKERS),
        "--samples",
        str(SAMPLES),
        "--profile-windows",
        str(PROFILE),
        "--train-fit-ref",
        str(TRAIN_FIT),
        "--print-json",
    ]
    # Capture via summarize_paths; CLI wiring is covered by parse + summarize_paths.
    summary = f0eff.summarize_paths(
        MARKERS, SAMPLES, profile_windows_path=PROFILE, train_fit_ref_path=TRAIN_FIT
    )
    require(summary["coverage"]["marker_count"] == 2, "CLI path should see both markers")
    args = f0eff.parse_args(argv)
    require(args.print_json is True, "parse --print-json")
    require(args.train_fit_ref == TRAIN_FIT, "parse train-fit ref")


CASES = (
    ("minicpm5_machine_readable", check_minicpm5_machine_readable),
    ("peak_vram_matches_samples", check_peak_vram_matches_samples),
    ("tokens_from_agoge_not_gpu", check_tokens_from_agoge_not_gpu),
    ("energy_skips_long_gap", check_energy_skips_long_gap),
    ("energy_trapezoid_regular", check_energy_trapezoid_regular),
    ("missing_power_is_not_zero", check_missing_power_is_not_zero),
    ("missing_counts", check_missing_counts),
    ("granite_same_formulas", check_granite_same_formulas),
    ("train_fit_not_overwritten", check_train_fit_not_overwritten),
    ("phase_and_profile", check_phase_and_profile),
    ("cli_minicpm5", check_cli_minicpm5),
)


def main(argv: list[str]) -> int:
    del argv
    failed = 0
    for name, fn in CASES:
        try:
            fn()
        except (CheckError, f0corr.SchemaError, f0eff.SummaryError, OSError, KeyError, TypeError, ValueError) as error:
            print(f"FAIL {name}: {error}", file=sys.stderr)
            failed += 1
            continue
        print(f"ok  {name}")
    if failed:
        print(f"{failed} F0 efficiency checks failed", file=sys.stderr)
        return 1
    print(f"{len(CASES)} F0 efficiency checks ok")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
