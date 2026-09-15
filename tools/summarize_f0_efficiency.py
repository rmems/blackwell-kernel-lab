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
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import check_f0_correlation as f0corr

SCHEMA_OUT = "bkl.f0_efficiency.v1"
HEADROOM_FLOOR_MIB = 2048
MAX_GAP_FACTOR = 3.0
NS_PER_S = 1_000_000_000
MS_TO_NS = 1_000_000
VRAM_AGREE_MIB = 1.0
PHYSICAL_NAMES = tuple(name for name, _unit in f0corr.PHYSICAL)
TRAIN_FIT_SOT = "docs/TRAIN_FIT_5080.md (#44 / RM-1053)"
ENERGY_NOTE = (
    "APPROXIMATE estimate from sampled board power (W), not laboratory-grade "
    "wall-power. Do not treat as datacenter energy efficiency."
)


class SummaryError(Exception):
    """Inputs could not be summarized."""


def r6(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


def measurement_status(obj: Any) -> str | None:
    if not isinstance(obj, dict):
        return None
    status = obj.get("status")
    if status in f0corr.MEASUREMENT_STATUS:
        return status
    return None


def ok_value(obj: Any) -> float | None:
    if measurement_status(obj) != "ok":
        return None
    value = obj.get("value")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(float(value)):
        return None
    return float(value)


def empty_status_counts() -> dict[str, int]:
    return {"ok": 0, "unavailable": 0, "unsupported": 0, "invalid": 0}


def tally_status(counts: dict[str, int], obj: Any) -> None:
    status = measurement_status(obj)
    if status is None:
        counts["invalid"] += 1
        return
    if status == "ok" and ok_value(obj) is None:
        counts["invalid"] += 1
        return
    counts[status] += 1


def collect_ok_values(samples: list[dict[str, Any]], field: str) -> tuple[list[float], dict[str, int]]:
    values: list[float] = []
    counts = empty_status_counts()
    for sample in samples:
        obj = sample.get(field)
        tally_status(counts, obj)
        value = ok_value(obj)
        if value is not None:
            values.append(value)
    return values, counts


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def extreme_or_none(values: list[float], fn: Any) -> float | None:
    if not values:
        return None
    return float(fn(values))


def sample_variance(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    return statistics.variance(values)


def missing_total(counts: dict[str, int]) -> int:
    return counts["unavailable"] + counts["unsupported"] + counts["invalid"]


def with_counts(payload: dict[str, Any], counts: dict[str, int]) -> dict[str, Any]:
    payload["ok_count"] = counts["ok"]
    payload["unavailable_count"] = counts["unavailable"]
    payload["unsupported_count"] = counts["unsupported"]
    payload["invalid_count"] = counts["invalid"]
    payload["missing_count"] = missing_total(counts)
    return payload


def counter_or_none(marker: dict[str, Any], key: str) -> int | None:
    value = marker.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def elapsed_s(first: dict[str, Any], last: dict[str, Any]) -> float | None:
    dt_ns = last["monotonic_ns"] - first["monotonic_ns"]
    if dt_ns <= 0:
        return None
    return dt_ns / NS_PER_S


def rate(delta: int | None, seconds: float | None) -> float | None:
    if delta is None or seconds is None or seconds <= 0 or delta < 0:
        return None
    return delta / seconds


def sorted_markers(markers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(markers, key=f0corr.marker_sort_key)


def sorted_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(samples, key=lambda item: (item["monotonic_ns"], item["timestamp_utc"]))


def unique_run_ids(rows: list[dict[str, Any]]) -> list[str]:
    seen: list[str] = []
    for row in rows:
        run_id = row.get("agoge_run_id")
        if isinstance(run_id, str) and run_id and run_id not in seen:
            seen.append(run_id)
    return seen


def filter_run(rows: list[dict[str, Any]], run_id: str) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("agoge_run_id") == run_id]


def cadence_ns(sample: dict[str, Any]) -> int | None:
    cadence_ms = sample.get("cadence_ms")
    if isinstance(cadence_ms, bool) or not isinstance(cadence_ms, int) or cadence_ms <= 0:
        return None
    return cadence_ms * MS_TO_NS


def pair_max_gap_ns(prev: dict[str, Any], cur: dict[str, Any], max_gap_factor: float) -> int | None:
    cadences = [item for item in (cadence_ns(prev), cadence_ns(cur)) if item is not None]
    if not cadences:
        return None
    return int(max_gap_factor * min(cadences))


def distribution(values: list[float], unit: str) -> dict[str, Any]:
    return {
        "unit": unit,
        "count": len(values),
        "min": r6(extreme_or_none(values, min)),
        "max": r6(extreme_or_none(values, max)),
        "mean": r6(mean_or_none(values)),
        "variance": r6(sample_variance(values)),
        "stdev": r6(None if len(values) < 2 else statistics.stdev(values)),
    }


def step_durations_s(markers: list[dict[str, Any]]) -> list[float]:
    ordered = sorted_markers(markers)
    durations: list[float] = []
    for prev, cur in zip(ordered, ordered[1:]):
        if cur["global_step"] <= prev["global_step"]:
            continue
        seconds = elapsed_s(prev, cur)
        if seconds is not None:
            durations.append(seconds)
    return durations


def counter_delta(first: dict[str, Any], last: dict[str, Any], key: str) -> int | None:
    start = counter_or_none(first, key)
    end = counter_or_none(last, key)
    if start is None or end is None or end < start:
        return None
    return end - start


def agoge_reported_throughput(markers: list[dict[str, Any]]) -> dict[str, Any] | None:
    ordered = sorted_markers(markers)
    if not ordered:
        return None
    last = ordered[-1].get("throughput_tokens_per_s")
    if not isinstance(last, dict):
        return None
    return {
        "value": r6(ok_value(last)),
        "unit": last.get("unit"),
        "status": last.get("status"),
        "note": "Agoge-owned optional field; not used to derive tokens/s here",
    }


def throughput_from_markers(markers: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted_markers(markers)
    payload: dict[str, Any] = {
        "source": "agoge_counters_and_monotonic_ns",
        "not_inferred_from_gpu_load": True,
        "marker_count": len(ordered),
        "tokens_accepted_delta": None,
        "examples_accepted_delta": None,
        "elapsed_s": None,
        "tokens_per_s": None,
        "examples_per_s": None,
        "agoge_reported_tokens_per_s": agoge_reported_throughput(ordered),
    }
    if len(ordered) < 2:
        return payload
    first, last = ordered[0], ordered[-1]
    payload["elapsed_s"] = r6(elapsed_s(first, last))
    payload["tokens_accepted_delta"] = counter_delta(first, last, "tokens_accepted")
    payload["examples_accepted_delta"] = counter_delta(first, last, "examples_accepted")
    payload["tokens_per_s"] = r6(rate(payload["tokens_accepted_delta"], payload["elapsed_s"]))
    payload["examples_per_s"] = r6(rate(payload["examples_accepted_delta"], payload["elapsed_s"]))
    return payload


def field_block(
    samples: list[dict[str, Any]],
    field: str,
    *,
    avg_key: str,
    peak_key: str,
    min_key: str | None,
    unit: str,
) -> dict[str, Any]:
    values, counts = collect_ok_values(samples, field)
    payload: dict[str, Any] = {"unit": unit, avg_key: r6(mean_or_none(values)), peak_key: r6(extreme_or_none(values, max))}
    if min_key is not None:
        payload[min_key] = r6(extreme_or_none(values, min))
    return with_counts(payload, counts)


def vram_block(samples: list[dict[str, Any]]) -> dict[str, Any]:
    used, used_counts = collect_ok_values(samples, "vram_used")
    free, free_counts = collect_ok_values(samples, "vram_free")
    headroom, headroom_counts = collect_ok_values(samples, "headroom")
    min_free = extreme_or_none(free, min)
    min_headroom = extreme_or_none(headroom, min)
    effective_headroom = min_headroom if min_headroom is not None else min_free
    meets_floor = None
    if effective_headroom is not None:
        meets_floor = effective_headroom >= HEADROOM_FLOOR_MIB
    return {
        "unit": "MiB",
        "peak_used_mib": r6(extreme_or_none(used, max)),
        "min_free_mib": r6(min_free),
        "min_headroom_mib": r6(min_headroom),
        "effective_min_headroom_mib": r6(effective_headroom),
        "headroom_floor_mib": HEADROOM_FLOOR_MIB,
        "meets_host_floor": meets_floor,
        "peak_used_counts": used_counts,
        "min_free_counts": free_counts,
        "min_headroom_counts": headroom_counts,
        "ok_count": used_counts["ok"],
        "unavailable_count": used_counts["unavailable"],
        "unsupported_count": used_counts["unsupported"],
        "invalid_count": used_counts["invalid"],
        "missing_count": missing_total(used_counts),
        "note": (
            "Peak/min are extrema of ok samples only. Missing ≠ zero. "
            f"{TRAIN_FIT_SOT} remains the concise train-fit summary."
        ),
    }


def field_inventory(samples: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    inventory: dict[str, dict[str, int]] = {}
    for name in PHYSICAL_NAMES:
        _values, counts = collect_ok_values(samples, name)
        inventory[name] = counts
    return inventory


def observed_intervals_ms(samples: list[dict[str, Any]]) -> list[float]:
    ordered = sorted_samples(samples)
    intervals: list[float] = []
    for prev, cur in zip(ordered, ordered[1:]):
        dt_ns = cur["monotonic_ns"] - prev["monotonic_ns"]
        if dt_ns > 0:
            intervals.append(dt_ns / 1e6)
    return intervals


def planned_cadences_ms(samples: list[dict[str, Any]]) -> list[int]:
    cadences: list[int] = []
    for sample in samples:
        cadence = sample.get("cadence_ms")
        if isinstance(cadence, int) and not isinstance(cadence, bool) and cadence > 0:
            cadences.append(cadence)
    return cadences


def expected_sample_count(samples: list[dict[str, Any]]) -> int | None:
    ordered = sorted_samples(samples)
    if not ordered:
        return None
    if len(ordered) == 1:
        return 1
    cadences = planned_cadences_ms(ordered)
    if not cadences:
        return None
    cadence_med = int(statistics.median(cadences))
    span_ns = ordered[-1]["monotonic_ns"] - ordered[0]["monotonic_ns"]
    return 1 + math.floor(span_ns / (cadence_med * MS_TO_NS))


def coverage_block(
    markers: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    long_gaps: list[dict[str, Any]],
) -> dict[str, Any]:
    expected = expected_sample_count(samples)
    observed = len(samples)
    ratio = None
    if expected is not None and expected > 0:
        ratio = observed / expected
    cadences = planned_cadences_ms(samples)
    intervals = observed_intervals_ms(samples)
    return {
        "marker_count": len(markers),
        "sample_count": observed,
        "joined_count": len(pairs),
        "unjoined_count": observed - len(pairs),
        "planned_cadence_ms": sorted(set(cadences)),
        "observed_median_interval_ms": r6(None if not intervals else statistics.median(intervals)),
        "expected_samples": expected,
        "coverage_ratio": r6(ratio),
        "long_gap_count": len(long_gaps),
        "long_gaps": long_gaps,
        "missing_sample_policy": (
            "Counts preserve unavailable/unsupported/invalid per field. "
            "Energy does not interpolate across gaps > max_gap_factor × cadence "
            "or pairs with missing power. Missing power is never treated as 0 W."
        ),
    }


def classify_power_pair(
    prev: dict[str, Any],
    cur: dict[str, Any],
    max_gap_factor: float,
) -> tuple[str, int, float | None, float | None]:
    dt_ns = cur["monotonic_ns"] - prev["monotonic_ns"]
    p0 = ok_value(prev.get("power"))
    p1 = ok_value(cur.get("power"))
    if dt_ns <= 0:
        return ("non_increasing", dt_ns, p0, p1)
    if p0 is None or p1 is None:
        return ("missing_power", dt_ns, p0, p1)
    allowed = pair_max_gap_ns(prev, cur, max_gap_factor)
    if allowed is None or dt_ns > allowed:
        return ("long_gap", dt_ns, p0, p1)
    return ("integrate", dt_ns, p0, p1)


def long_gap_record(prev: dict[str, Any], cur: dict[str, Any], dt_ns: int, max_gap_factor: float) -> dict[str, Any]:
    allowed = pair_max_gap_ns(prev, cur, max_gap_factor)
    return {
        "from_monotonic_ns": prev["monotonic_ns"],
        "to_monotonic_ns": cur["monotonic_ns"],
        "dt_ms": r6(dt_ns / 1e6),
        "max_allowed_ms": None if allowed is None else r6(allowed / 1e6),
        "reason": "no_cadence" if allowed is None else "dt_exceeds_max_gap_factor",
    }


def integrate_board_power(samples: list[dict[str, Any]], max_gap_factor: float) -> dict[str, Any]:
    ordered = sorted_samples(samples)
    joules = 0.0
    span_s = 0.0
    integrated = 0
    skipped_gap = 0
    skipped_power = 0
    skipped_order = 0
    long_gaps: list[dict[str, Any]] = []
    for prev, cur in zip(ordered, ordered[1:]):
        kind, dt_ns, p0, p1 = classify_power_pair(prev, cur, max_gap_factor)
        if kind == "non_increasing":
            skipped_order += 1
            continue
        if kind == "missing_power":
            skipped_power += 1
            continue
        if kind == "long_gap":
            skipped_gap += 1
            long_gaps.append(long_gap_record(prev, cur, dt_ns, max_gap_factor))
            continue
        dt_s = dt_ns / NS_PER_S
        joules += 0.5 * (p0 + p1) * dt_s
        span_s += dt_s
        integrated += 1
    total = joules if integrated else None
    return {
        "label": "approximate",
        "method": "trapezoidal_board_power",
        "max_gap_factor": max_gap_factor,
        "pairs_integrated": integrated,
        "pairs_skipped_long_gap": skipped_gap,
        "pairs_skipped_missing_power": skipped_power,
        "pairs_skipped_non_increasing": skipped_order,
        "integrated_span_s": r6(span_s if integrated else None),
        "approximate_joules": r6(total),
        "long_gaps": long_gaps,
        "note": ENERGY_NOTE,
    }


def energy_rates(energy: dict[str, Any], throughput: dict[str, Any], step_count: int) -> dict[str, Any]:
    joules = energy.get("approximate_joules")
    tokens = throughput.get("tokens_accepted_delta")
    per_step = None
    per_token = None
    if joules is not None and step_count > 0:
        per_step = joules / step_count
    if joules is not None and isinstance(tokens, int) and tokens > 0:
        per_token = joules / tokens
    energy["approximate_joules_per_step"] = r6(per_step)
    energy["approximate_joules_per_token"] = r6(per_token)
    energy["step_count_for_energy"] = step_count
    energy["token_count_for_energy"] = tokens
    return energy


def samples_by_phase(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for marker, sample in pairs:
        buckets[str(marker["phase"])].append(sample)
    return dict(buckets)


def phase_metrics(samples: list[dict[str, Any]]) -> dict[str, Any]:
    gpu = field_block(samples, "utilization_gpu", avg_key="avg", peak_key="peak", min_key=None, unit="%")
    mem = field_block(samples, "utilization_memory", avg_key="avg", peak_key="peak", min_key=None, unit="%")
    power = field_block(samples, "power", avg_key="avg_w", peak_key="peak_w", min_key=None, unit="W")
    temp = field_block(
        samples, "temperature_gpu", avg_key="avg_c", peak_key="peak_c", min_key=None, unit="C"
    )
    vram = vram_block(samples)
    return {
        "sample_count": len(samples),
        "utilization_gpu": gpu,
        "utilization_memory": mem,
        "power": power,
        "temperature_gpu": temp,
        "vram": {
            "peak_used_mib": vram["peak_used_mib"],
            "min_free_mib": vram["min_free_mib"],
            "min_headroom_mib": vram["min_headroom_mib"],
            "ok_count": vram["ok_count"],
            "missing_count": vram["missing_count"],
        },
    }


def load_optional_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    return f0corr.load_jsonl(path)


def load_optional_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    obj = json.loads(path.read_text())
    if not isinstance(obj, dict):
        raise SummaryError(f"{path}: expected a JSON object")
    return obj


def compact_profile_record(row: dict[str, Any]) -> dict[str, Any]:
    ref = row.get("profile_window_ref")
    f0corr.require_nonempty_str(ref, "profile_window_ref must be a non-empty string")
    return {
        "profile_window_ref": ref,
        "agoge_run_id": row.get("agoge_run_id"),
        "backend": row.get("backend"),
        "status": row.get("status"),
        "runtime_ms": row.get("runtime_ms"),
        "kernel_count": row.get("kernel_count"),
        "sync_stall_ms": row.get("sync_stall_ms"),
        "h2d_ms": row.get("h2d_ms"),
        "d2h_ms": row.get("d2h_ms"),
        "top_kernels": row.get("top_kernels"),
        "note": row.get("note"),
    }


def index_profile_windows(
    rows: list[dict[str, Any]], run_id: str
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("agoge_run_id") not in (None, run_id):
            continue
        record = compact_profile_record(row)
        indexed[record["profile_window_ref"]] = record
    return indexed


def window_span_s(samples: list[dict[str, Any]]) -> float | None:
    ordered = sorted_samples(samples)
    if len(ordered) < 2:
        return None
    return elapsed_s(ordered[0], ordered[-1])


def samples_by_profile_ref(samples: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_ref: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        ref = sample.get("profile_window_ref")
        if isinstance(ref, str) and ref:
            by_ref[ref].append(sample)
    return dict(by_ref)


def joined_phases_for_window(
    window_samples: list[dict[str, Any]],
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
) -> list[str]:
    phase_by_sample = {id(sample): marker.get("phase") for marker, sample in pairs}
    phases = {
        str(phase_by_sample[id(sample)])
        for sample in window_samples
        if id(sample) in phase_by_sample
    }
    return sorted(phases)


def sampled_profile_window(
    ref: str,
    window_samples: list[dict[str, Any]],
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    compact: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    power_vals, power_counts = collect_ok_values(window_samples, "power")
    extra = compact.get(ref) or None
    return {
        "profile_window_ref": ref,
        "sample_count": len(window_samples),
        "joined_phases": joined_phases_for_window(window_samples, pairs),
        "span_s": r6(window_span_s(window_samples)),
        "power_avg_w": r6(mean_or_none(power_vals)),
        "power_ok_count": power_counts["ok"],
        "power_missing_count": missing_total(power_counts),
        "compact_summary": extra,
        "note": (
            "Window identity comes from bkl_gpu_sample.profile_window_ref. "
            "Kernel/runtime fields come from optional #54 compact summaries."
        ),
    }


def orphan_profile_window(ref: str, extra: dict[str, Any]) -> dict[str, Any]:
    attached = dict(extra)
    attached["note"] = "Compact #54 summary with no matching GPU samples"
    return {
        "profile_window_ref": ref,
        "sample_count": 0,
        "joined_phases": [],
        "span_s": None,
        "power_avg_w": None,
        "power_ok_count": 0,
        "power_missing_count": 0,
        "compact_summary": attached,
        "note": attached["note"],
    }


def profile_window_summaries(
    samples: list[dict[str, Any]],
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    compact_rows: list[dict[str, Any]],
    run_id: str,
) -> list[dict[str, Any]]:
    by_ref = samples_by_profile_ref(samples)
    compact = index_profile_windows(compact_rows, run_id)
    summaries = [
        sampled_profile_window(ref, by_ref[ref], pairs, compact) for ref in sorted(by_ref)
    ]
    summaries.extend(
        orphan_profile_window(ref, compact[ref]) for ref in sorted(compact) if ref not in by_ref
    )
    return summaries


def compare_extrema(
    derived: float | None, reference: float | None, tol: float
) -> dict[str, Any]:
    if derived is None or reference is None:
        return {"agreement": None, "delta": None, "reason": "one_or_both_values_missing"}
    delta = derived - reference
    return {
        "agreement": abs(delta) <= tol,
        "delta": r6(delta),
        "tolerance": tol,
        "reason": None if abs(delta) <= tol else "outside_tolerance",
    }


def train_fit_comparison(vram: dict[str, Any], reference: dict[str, Any] | None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "source_of_truth": TRAIN_FIT_SOT,
        "derived_is_evidence_only": True,
        "comparable_peak": False,
        "comparable_headroom": False,
        "peak_vram": None,
        "min_headroom": None,
        "note": (
            "Do not replace the #44 train-fit table with this derivation. "
            "Trainer allocated bytes and sampler vram_used are different bases."
        ),
    }
    if reference is None:
        payload["reference"] = None
        return payload
    payload["reference"] = reference
    derived_peak = vram.get("peak_used_mib")
    ref_peak = reference.get("peak_allocated_mib")
    ref_peak_basis = reference.get("peak_basis")
    payload["comparable_peak"] = ref_peak_basis == "bkl_gpu_sample.vram_used"
    if payload["comparable_peak"]:
        payload["peak_vram"] = compare_extrema(derived_peak, ref_peak, VRAM_AGREE_MIB)
    else:
        payload["peak_vram"] = {
            "derived_peak_used_mib": derived_peak,
            "reference_peak_mib": ref_peak,
            "reference_basis": ref_peak_basis,
            "agreement": None,
            "reason": "different_measurement_basis",
        }
    derived_headroom = vram.get("effective_min_headroom_mib")
    ref_min_free = reference.get("min_free_mib")
    payload["comparable_headroom"] = reference.get("min_free_basis") == "bkl_gpu_sample.vram_free"
    if payload["comparable_headroom"]:
        payload["min_headroom"] = compare_extrema(derived_headroom, ref_min_free, VRAM_AGREE_MIB)
    else:
        payload["min_headroom"] = {
            "derived_effective_min_headroom_mib": derived_headroom,
            "reference_min_free_mib": ref_min_free,
            "agreement": None,
            "reason": "train_fit_min_free_not_on_sampler_basis_or_missing",
        }
    return payload


def run_identity(markers: list[dict[str, Any]], samples: list[dict[str, Any]], run_id: str) -> dict[str, Any]:
    source = markers[0] if markers else samples[0]
    model_id = None
    model_revision = None
    dataset = None
    if markers:
        model_id = markers[0].get("model_id")
        model_revision = markers[0].get("model_revision")
        dataset = markers[0].get("dataset")
    return {
        "agoge_run_id": run_id,
        "model_id": model_id,
        "model_revision": model_revision,
        "dataset": dataset,
        "host": source.get("host"),
        "gpu": source.get("gpu"),
        "formulas_are_model_agnostic": True,
    }


def unjoined_samples(
    samples: list[dict[str, Any]], pairs: list[tuple[dict[str, Any], dict[str, Any]]]
) -> list[dict[str, Any]]:
    joined_ids = {id(sample) for _marker, sample in pairs}
    return [sample for sample in samples if id(sample) not in joined_ids]


def resolve_run_id(markers: list[dict[str, Any]], samples: list[dict[str, Any]], requested: str | None) -> str:
    ids = unique_run_ids(markers + samples)
    f0corr.require(ids, "no agoge_run_id present")
    if requested is not None:
        f0corr.require(requested in ids, f"run id {requested!r} not in bundle ({ids})")
        return requested
    f0corr.require(len(ids) == 1, f"multiple run ids {ids}; pass --run-id")
    return ids[0]


def validate_bundle(markers: list[dict[str, Any]], samples: list[dict[str, Any]]) -> None:
    f0corr.require(markers or samples, "need at least one marker or GPU sample")
    for rec in markers:
        f0corr.check_marker(rec)
    for rec in samples:
        f0corr.check_sample(rec)


def summarize_run(
    markers: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    *,
    max_gap_factor: float,
    profile_rows: list[dict[str, Any]],
    train_fit_ref: dict[str, Any] | None,
) -> dict[str, Any]:
    pairs = f0corr.join_samples(markers, samples)
    throughput = throughput_from_markers(markers)
    durations = step_durations_s(markers)
    energy = integrate_board_power(samples, max_gap_factor)
    energy = energy_rates(energy, throughput, len(durations))
    vram = vram_block(samples)
    power = field_block(samples, "power", avg_key="avg_w", peak_key="peak_w", min_key=None, unit="W")
    temp = field_block(
        samples, "temperature_gpu", avg_key="avg_c", peak_key="peak_c", min_key=None, unit="C"
    )
    gpu_util = field_block(
        samples, "utilization_gpu", avg_key="avg", peak_key="peak", min_key=None, unit="%"
    )
    mem_util = field_block(
        samples, "utilization_memory", avg_key="avg", peak_key="peak", min_key=None, unit="%"
    )
    phases = {name: phase_metrics(group) for name, group in samples_by_phase(pairs).items()}
    long_gaps = energy["long_gaps"]
    return {
        "schema_version": SCHEMA_OUT,
        "energy_is_approximate": True,
        "hardware_note": (
            "RTX 5080 / sm_120 / ~16 GB. Leave ≥2 GiB free. "
            "Derived measurements, not a trainer or safety controller."
        ),
        "run": run_identity(markers, samples, markers[0]["agoge_run_id"] if markers else samples[0]["agoge_run_id"]),
        "coverage": coverage_block(markers, samples, pairs, long_gaps),
        "throughput": throughput,
        "step_time": distribution(durations, "s"),
        "vram": vram,
        "power": power,
        "temperature_gpu": temp,
        "utilization_gpu": gpu_util,
        "utilization_memory": mem_util,
        "energy": energy,
        "by_phase": phases,
        "unjoined_sample_count": len(unjoined_samples(samples, pairs)),
        "profile_windows": profile_window_summaries(
            samples,
            pairs,
            profile_rows,
            markers[0]["agoge_run_id"] if markers else samples[0]["agoge_run_id"],
        ),
        "field_status_counts": field_inventory(samples),
        "train_fit_comparison": train_fit_comparison(vram, train_fit_ref),
    }


def fmt_num(value: Any, unit: str = "", missing: str = "n/a") -> str:
    if value is None:
        return missing
    if unit:
        return f"{value} {unit}"
    return str(value)


def report_header(summary: dict[str, Any]) -> list[str]:
    run = summary["run"]
    return [
        f"# F0 efficiency summary (`{summary['schema_version']}`)",
        "",
        f"- Run: `{run.get('agoge_run_id')}`",
        f"- Model: `{run.get('model_id')}` (formulas are model-agnostic)",
        f"- Host: `{((run.get('host') or {}).get('hostname'))}`",
        "",
    ]


def report_throughput(summary: dict[str, Any]) -> list[str]:
    thr = summary["throughput"]
    step = summary["step_time"]
    return [
        "## Throughput (Agoge counters + monotonic_ns)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| tokens/s | {fmt_num(thr.get('tokens_per_s'), 'token/s')} |",
        f"| examples/s | {fmt_num(thr.get('examples_per_s'), 'example/s')} |",
        f"| elapsed | {fmt_num(thr.get('elapsed_s'), 's')} |",
        f"| tokens delta | {fmt_num(thr.get('tokens_accepted_delta'))} |",
        f"| step-time n / mean / stdev | {step['count']} / {fmt_num(step.get('mean'), 's')} / {fmt_num(step.get('stdev'), 's')} |",
        "",
        "Tokens/s and step-time are **not** inferred from GPU utilization.",
        "",
    ]


def report_vram(summary: dict[str, Any]) -> list[str]:
    vram = summary["vram"]
    fit = summary["train_fit_comparison"]
    return [
        "## VRAM / headroom (BKL samples)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| peak used | {fmt_num(vram.get('peak_used_mib'), 'MiB')} |",
        f"| min free | {fmt_num(vram.get('min_free_mib'), 'MiB')} |",
        f"| min headroom | {fmt_num(vram.get('min_headroom_mib'), 'MiB')} |",
        f"| host floor | {vram.get('headroom_floor_mib')} MiB (meets={vram.get('meets_host_floor')}) |",
        f"| ok / missing used samples | {vram.get('ok_count')} / {vram.get('missing_count')} |",
        "",
        f"Concise train-fit SoT remains **{fit['source_of_truth']}**. "
        "This summary is correlated evidence, not a second ceiling.",
        "",
    ]


def report_power_energy(summary: dict[str, Any]) -> list[str]:
    cov = summary["coverage"]
    power = summary["power"]
    temp = summary["temperature_gpu"]
    energy = summary["energy"]
    return [
        "## Power / thermal",
        "",
        "| Metric | Value | missing |",
        "|---|---|---|",
        f"| board power avg / peak | {fmt_num(power.get('avg_w'), 'W')} / {fmt_num(power.get('peak_w'), 'W')} | {power.get('missing_count')} |",
        f"| GPU temp peak | {fmt_num(temp.get('peak_c'), 'C')} | {temp.get('missing_count')} |",
        "",
        "## Energy (approximate)",
        "",
        energy["note"],
        "",
        "| Item | Value |",
        "|---|---|",
        f"| method | {energy.get('method')} |",
        f"| planned cadence | {fmt_num(cov.get('planned_cadence_ms'), 'ms')} |",
        f"| max gap factor | {energy.get('max_gap_factor')} × cadence |",
        f"| pairs integrated / long-gap / missing-power | {energy.get('pairs_integrated')} / {energy.get('pairs_skipped_long_gap')} / {energy.get('pairs_skipped_missing_power')} |",
        f"| approximate joules | {fmt_num(energy.get('approximate_joules'), 'J')} |",
        f"| approximate J/step | {fmt_num(energy.get('approximate_joules_per_step'), 'J/step')} |",
        f"| approximate J/token | {fmt_num(energy.get('approximate_joules_per_token'), 'J/token')} |",
        "",
        "## Coverage",
        "",
        f"- samples {cov.get('sample_count')}, expected {fmt_num(cov.get('expected_samples'))}, "
        f"coverage_ratio {fmt_num(cov.get('coverage_ratio'))}",
        f"- joined {cov.get('joined_count')}, unjoined {cov.get('unjoined_count')}, long gaps {cov.get('long_gap_count')}",
        f"- policy: {cov.get('missing_sample_policy')}",
        "",
    ]


def report_phases(summary: dict[str, Any]) -> list[str]:
    lines = ["## By phase (joined samples only)", ""]
    by_phase = summary.get("by_phase") or {}
    if not by_phase:
        lines.append("_No joined samples; phases are not inferred from GPU load._")
        lines.append("")
        return lines
    for name in sorted(by_phase):
        block = by_phase[name]
        gpu = block["utilization_gpu"]
        mem = block["utilization_memory"]
        lines.append(
            f"- `{name}`: n={block['sample_count']}; GPU util avg {fmt_num(gpu.get('avg'), '%')}; "
            f"mem util avg {fmt_num(mem.get('avg'), '%')}; "
            f"peak VRAM {fmt_num(block['vram'].get('peak_used_mib'), 'MiB')}"
        )
    return lines


def report_profiles(summary: dict[str, Any]) -> list[str]:
    windows = summary.get("profile_windows") or []
    if not windows:
        return []
    lines = ["", "## Profiler windows", ""]
    for window in windows:
        compact = window.get("compact_summary") or {}
        lines.append(
            f"- `{window['profile_window_ref']}`: samples={window['sample_count']}, "
            f"phases={window['joined_phases']}, runtime_ms={compact.get('runtime_ms')}"
        )
    return lines


def report_lines(summary: dict[str, Any]) -> list[str]:
    lines = report_header(summary)
    lines.extend(report_throughput(summary))
    lines.extend(report_vram(summary))
    lines.extend(report_power_energy(summary))
    lines.extend(report_phases(summary))
    lines.extend(report_profiles(summary))
    return lines


def write_json(path: Path | None, summary: dict[str, Any]) -> str:
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return text


def write_report(path: Path | None, summary: dict[str, Any]) -> str:
    text = "\n".join(report_lines(summary)) + "\n"
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return text


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
    markers = f0corr.load_jsonl(markers_path)
    samples = f0corr.load_jsonl(samples_path)
    validate_bundle(markers, samples)
    chosen = resolve_run_id(markers, samples, run_id)
    markers = filter_run(markers, chosen)
    samples = filter_run(samples, chosen)
    profile_rows = load_optional_jsonl(profile_windows_path)
    train_fit_ref = load_optional_json(train_fit_ref_path)
    return summarize_run(
        markers,
        samples,
        max_gap_factor=max_gap_factor,
        profile_rows=profile_rows,
        train_fit_ref=train_fit_ref,
    )


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        f0corr.require(args.max_gap_factor > 0, "--max-gap-factor must be > 0")
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
