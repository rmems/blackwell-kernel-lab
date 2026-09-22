"""Device identity, profiler windows, and train-fit comparison for F0 summaries."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import check_f0_correlation as f0corr
from f0_efficiency_numbers import (
    TRAIN_FIT_SOT,
    VRAM_AGREE_MIB,
    SummaryError,
    collect_ok_values,
    comparable_float,
    mean_or_none,
    missing_total,
    r6,
    sorted_samples,
    unique_run_ids,
    elapsed_s,
)


def _reject_nonfinite_json(token: str) -> None:
    raise SummaryError(f"non-finite JSON number {token!r} is not allowed")


def load_optional_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    return f0corr.load_jsonl(path)


def load_optional_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        obj = json.loads(path.read_text(), parse_constant=_reject_nonfinite_json)
    except json.JSONDecodeError as error:
        raise SummaryError(f"{path}: {error}") from error
    if not isinstance(obj, dict):
        raise SummaryError(f"{path}: expected a JSON object")
    return obj


def selected_host_gpu(
    markers: list[dict[str, Any]], samples: list[dict[str, Any]]
) -> tuple[Any, Any]:
    source = markers[0] if markers else samples[0]
    return source.get("host"), source.get("gpu")


def sample_matches_selected_device(sample: dict[str, Any], host: Any, gpu: Any) -> bool:
    sample_host = sample.get("host")
    if not isinstance(host, dict) or not isinstance(sample_host, dict):
        return False
    if sample_host.get("hostname") != host.get("hostname"):
        return False
    return f0corr.gpu_identities_match(gpu, sample.get("gpu"))


def partition_physical_samples(
    samples: list[dict[str, Any]], host: Any, gpu: Any
) -> tuple[list[dict[str, Any]], int]:
    matched = [sample for sample in samples if sample_matches_selected_device(sample, host, gpu)]
    return matched, len(samples) - len(matched)


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


def index_profile_windows(rows: list[dict[str, Any]], run_id: str) -> dict[str, dict[str, Any]]:
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
    summaries = [sampled_profile_window(ref, by_ref[ref], pairs, compact) for ref in sorted(by_ref)]
    summaries.extend(orphan_profile_window(ref, compact[ref]) for ref in sorted(compact) if ref not in by_ref)
    return summaries


def compare_extrema(derived: Any, reference: Any, tol: float) -> dict[str, Any]:
    derived_num = comparable_float(derived)
    reference_num = comparable_float(reference)
    if derived_num is None or reference_num is None:
        return {"agreement": None, "delta": None, "reason": "one_or_both_values_missing"}
    delta = derived_num - reference_num
    return {
        "agreement": abs(delta) <= tol,
        "delta": r6(delta),
        "tolerance": tol,
        "reason": None if abs(delta) <= tol else "outside_tolerance",
    }


def derived_min_for_basis(vram: dict[str, Any], basis: Any) -> float | None:
    if basis == "bkl_gpu_sample.vram_free":
        value = vram.get("min_free_mib")
    else:
        value = vram.get("effective_min_headroom_mib")
    return comparable_float(value)


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
    ref_min_free = reference.get("min_free_mib")
    min_free_basis = reference.get("min_free_basis")
    payload["comparable_headroom"] = min_free_basis == "bkl_gpu_sample.vram_free"
    derived_min = derived_min_for_basis(vram, min_free_basis)
    if payload["comparable_headroom"]:
        payload["min_headroom"] = compare_extrema(derived_min, ref_min_free, VRAM_AGREE_MIB)
    else:
        payload["min_headroom"] = {
            "derived_effective_min_headroom_mib": vram.get("effective_min_headroom_mib"),
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
