"""Numeric F0 efficiency derivations: throughput, VRAM, coverage, energy."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Any

import check_f0_correlation as f0corr
from f0_measurements import MEASUREMENT_STATUS

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
ENERGY_WINDOW_MISMATCH = "energy_window_mismatch"


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
    if status in MEASUREMENT_STATUS:
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


def require_max_gap_factor(value: float) -> None:
    if not math.isfinite(value) or value <= 0:
        raise SummaryError("--max-gap-factor must be a finite value > 0")


def pair_max_gap_ns(prev: dict[str, Any], cur: dict[str, Any], max_gap_factor: float) -> int | None:
    if not math.isfinite(max_gap_factor) or max_gap_factor <= 0:
        return None
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


def sample_coalesced_headroom(sample: dict[str, Any]) -> float | None:
    headroom = ok_value(sample.get("headroom"))
    if headroom is not None:
        return headroom
    return ok_value(sample.get("vram_free"))


def coalesced_headroom_values(samples: list[dict[str, Any]]) -> list[float]:
    return [value for sample in samples if (value := sample_coalesced_headroom(sample)) is not None]


def vram_block(samples: list[dict[str, Any]]) -> dict[str, Any]:
    used, used_counts = collect_ok_values(samples, "vram_used")
    free, free_counts = collect_ok_values(samples, "vram_free")
    headroom, headroom_counts = collect_ok_values(samples, "headroom")
    min_free = extreme_or_none(free, min)
    min_headroom = extreme_or_none(headroom, min)
    effective_headroom = extreme_or_none(coalesced_headroom_values(samples), min)
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
            "Effective headroom coalesces ok headroom, else ok vram_free, per sample. "
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
    cadence_med = statistics.median(cadences)
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
            "or pairs with missing power. Missing power is never treated as 0 W. "
            "Cadence gaps are recorded even when a power endpoint is missing."
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
    allowed = pair_max_gap_ns(prev, cur, max_gap_factor)
    long_gap = allowed is None or dt_ns > allowed
    missing = p0 is None or p1 is None
    if long_gap and missing:
        return ("long_gap_missing_power", dt_ns, p0, p1)
    if missing:
        return ("missing_power", dt_ns, p0, p1)
    if long_gap:
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
    require_max_gap_factor(max_gap_factor)
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
        if "long_gap" in kind:
            skipped_gap += 1
            long_gaps.append(long_gap_record(prev, cur, dt_ns, max_gap_factor))
        if "missing_power" in kind:
            skipped_power += 1
            continue
        if kind == "long_gap":
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


def marker_window_ns(markers: list[dict[str, Any]]) -> tuple[int, int] | None:
    ordered = sorted_markers(markers)
    if len(ordered) < 2:
        return None
    return ordered[0]["monotonic_ns"], ordered[-1]["monotonic_ns"]


def energy_has_complete_integral(energy: dict[str, Any]) -> bool:
    if energy.get("approximate_joules") is None:
        return False
    return not (energy.get("pairs_skipped_long_gap") or energy.get("pairs_skipped_missing_power"))


def samples_inside_marker_window(samples: list[dict[str, Any]], window: tuple[int, int]) -> bool:
    start_ns, end_ns = window
    return all(start_ns <= sample["monotonic_ns"] <= end_ns for sample in samples)


def spans_match(elapsed: Any, span: Any) -> bool:
    if not isinstance(elapsed, (int, float)) or not isinstance(span, (int, float)):
        return False
    if isinstance(elapsed, bool) or isinstance(span, bool):
        return False
    return math.isclose(float(elapsed), float(span), rel_tol=0.0, abs_tol=1e-6)


def energy_window_matches_counters(
    energy: dict[str, Any],
    throughput: dict[str, Any],
    markers: list[dict[str, Any]] | None,
    samples: list[dict[str, Any]] | None,
) -> bool:
    window = marker_window_ns(markers or [])
    if not energy_has_complete_integral(energy) or window is None:
        return False
    if not samples_inside_marker_window(samples or [], window):
        return False
    return spans_match(throughput.get("elapsed_s"), energy.get("integrated_span_s"))


def aligned_quotient(joules: Any, aligned: bool, denominator: Any) -> float | None:
    if not aligned or not isinstance(joules, (int, float)) or isinstance(joules, bool):
        return None
    if not isinstance(denominator, (int, float)) or isinstance(denominator, bool) or denominator <= 0:
        return None
    return float(joules) / float(denominator)


def energy_rates(
    energy: dict[str, Any],
    throughput: dict[str, Any],
    step_count: int,
    *,
    markers: list[dict[str, Any]] | None = None,
    samples: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    joules = energy.get("approximate_joules")
    aligned = energy_window_matches_counters(energy, throughput, markers, samples)
    energy["approximate_joules_per_step"] = r6(aligned_quotient(joules, aligned, step_count))
    energy["approximate_joules_per_token"] = r6(
        aligned_quotient(joules, aligned, throughput.get("tokens_accepted_delta"))
    )
    energy["step_count_for_energy"] = step_count
    energy["token_count_for_energy"] = throughput.get("tokens_accepted_delta")
    energy["rates_omitted_reason"] = None if aligned else ENERGY_WINDOW_MISMATCH
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
    temp = field_block(samples, "temperature_gpu", avg_key="avg_c", peak_key="peak_c", min_key=None, unit="C")
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
