#!/usr/bin/env python3
"""Validate and join F0 Agoge ↔ BKL correlation fixtures (CPU, no GPU).

See docs/f0-correlation-schema.md. Used by .github/workflows/ci-cpu.yml.

Usage:
    python3 tools/check_f0_correlation.py \\
      --markers fixtures/f0-correlation/agoge-markers.jsonl \\
      --samples fixtures/f0-correlation/bkl-gpu-samples.jsonl \\
      --clock-skew fixtures/f0-correlation/clock-skew.json
"""

from __future__ import annotations

import argparse
import bisect
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import math

import f0_clock
from f0_measurements import (
    MARKER,
    PERCENT_FIELDS,
    PHYSICAL,
    SAMPLE,
    SCHEMA,
    SchemaError,
    check_measurement,
    check_ok_numeric,
    check_percent_bounds,
    check_throttle,
    parse_rfc3339_utc,
    require,
    require_capability_digest,
    require_nonempty_str,
    require_nonneg_int,
    require_optional_nonempty_str,
    require_optional_nonneg_int,
    require_positive_int,
)

FIXTURE_IDLE_MONOTONIC_NS = 1_500_000_000


def _reject_nonfinite_json(token: str) -> None:
    raise SchemaError(f"non-finite JSON number {token!r} is not allowed")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    text = path.read_text()
    for line_no, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            obj = json.loads(stripped, parse_constant=_reject_nonfinite_json)
        except json.JSONDecodeError as error:
            raise SchemaError(f"{path}:{line_no}: {error}") from error
        require(isinstance(obj, dict), f"{path}:{line_no}: expected object")
        rows.append(obj)
    require(rows, f"{path}: no records")
    return rows


def check_host(rec: dict[str, Any]) -> None:
    host = rec.get("host")
    require(isinstance(host, dict), "host object required")
    require_nonempty_str(host.get("hostname"), "host.hostname required")


def check_time(rec: dict[str, Any]) -> None:
    parse_rfc3339_utc(rec.get("timestamp_utc"))
    require_nonneg_int(rec.get("monotonic_ns"), "monotonic_ns must be a non-negative int")


def check_collector(rec: dict[str, Any]) -> None:
    collector = rec.get("collector")
    require(isinstance(collector, dict), "collector object required")
    require_nonempty_str(collector.get("id"), "collector.id must be a non-empty string")
    require_nonempty_str(collector.get("version"), "collector.version must be a non-empty string")


def check_envelope(rec: dict[str, Any], kind: str) -> None:
    require(rec.get("schema_version") == SCHEMA, f"unexpected schema_version: {rec.get('schema_version')}")
    require(rec.get("record_kind") == kind, f"expected record_kind {kind}, got {rec.get('record_kind')}")
    require_nonempty_str(rec.get("agoge_run_id"), "agoge_run_id required")
    check_host(rec)
    require(isinstance(rec.get("gpu"), dict), "gpu identity object required")
    check_time(rec)
    check_collector(rec)


def check_dataset(rec: dict[str, Any]) -> None:
    dataset = rec.get("dataset")
    require(isinstance(dataset, dict), "dataset object required")
    require_nonempty_str(dataset.get("id"), "dataset.id required")
    require_nonempty_str(dataset.get("split"), "dataset.split required")
    require_optional_nonempty_str(
        dataset.get("config_digest"),
        "dataset.config_digest must be null or a non-empty string",
    )


def check_marker_model(rec: dict[str, Any]) -> None:
    require(rec.get("cadence_ms") is None, "agoge_marker cadence_ms must be null")
    require_nonempty_str(rec.get("model_id"), "model_id required")
    require_nonempty_str(rec.get("model_revision"), "model_revision required")


def check_marker_step(rec: dict[str, Any]) -> None:
    require_nonempty_str(rec.get("phase"), "phase required")
    require_nonneg_int(rec.get("global_step"), "global_step must be a non-negative int")
    require_optional_nonneg_int(rec.get("microstep"), "microstep must be null or a non-negative int")
    require_optional_nonneg_int(
        rec.get("tokens_accepted"),
        "tokens_accepted must be null or a non-negative int",
    )
    require_optional_nonneg_int(
        rec.get("examples_accepted"),
        "examples_accepted must be null or a non-negative int",
    )


def check_marker_metrics(rec: dict[str, Any]) -> None:
    if "loss" in rec:
        check_measurement(rec["loss"], "loss", "1")
    if "throughput_tokens_per_s" in rec:
        check_measurement(rec["throughput_tokens_per_s"], "throughput_tokens_per_s", "token/s")


def check_marker(rec: dict[str, Any]) -> None:
    check_envelope(rec, MARKER)
    check_marker_model(rec)
    check_dataset(rec)
    check_marker_step(rec)
    check_marker_metrics(rec)


def check_sample_physical(rec: dict[str, Any]) -> None:
    for name, unit in PHYSICAL:
        require(name in rec, f"missing physical field {name}")
        check_measurement(rec[name], name, unit)
        if name in PERCENT_FIELDS:
            check_percent_bounds(rec[name], name)


def check_cuda(rec: dict[str, Any]) -> None:
    cuda = rec.get("cuda")
    require(isinstance(cuda, dict), "cuda object required")
    require_optional_nonempty_str(cuda.get("driver_version"), "cuda.driver_version must be null or string")
    require_optional_nonempty_str(cuda.get("runtime_version"), "cuda.runtime_version must be null or string")
    require_optional_nonempty_str(cuda.get("tool"), "cuda.tool must be null or string")


def check_sample(rec: dict[str, Any]) -> None:
    check_envelope(rec, SAMPLE)
    require(
        gpu_join_token(rec.get("gpu")) is not None,
        "bkl_gpu_sample requires a non-empty gpu.uuid or gpu.pci_bus_id",
    )
    require_positive_int(rec.get("cadence_ms"), "bkl_gpu_sample cadence_ms must be a positive int")
    check_sample_physical(rec)
    check_throttle(rec.get("throttle"))
    require_optional_nonempty_str(
        rec.get("profile_window_ref"),
        "profile_window_ref must be null or a non-empty string",
    )
    check_cuda(rec)
    if "capability_digest" in rec:
        require_capability_digest(
            rec.get("capability_digest"),
            "capability_digest must be sha256:<64 lowercase hex>",
        )


def nonempty_id(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def gpu_join_token(gpu: Any) -> tuple[str, str] | None:
    require(isinstance(gpu, dict), "gpu identity object required")
    uuid = nonempty_id(gpu.get("uuid"))
    if uuid is not None:
        return ("uuid", uuid)
    pci = nonempty_id(gpu.get("pci_bus_id"))
    if pci is not None:
        return ("pci", pci)
    return None


def gpu_identities_match(marker_gpu: Any, sample_gpu: Any) -> bool:
    if not isinstance(marker_gpu, dict) or not isinstance(sample_gpu, dict):
        return False
    marker_uuid = nonempty_id(marker_gpu.get("uuid"))
    sample_uuid = nonempty_id(sample_gpu.get("uuid"))
    if marker_uuid is not None and sample_uuid is not None:
        return marker_uuid == sample_uuid
    marker_pci = nonempty_id(marker_gpu.get("pci_bus_id"))
    sample_pci = nonempty_id(sample_gpu.get("pci_bus_id"))
    if marker_pci is not None and sample_pci is not None:
        return marker_pci == sample_pci
    return False


def join_bucket_key(rec: dict[str, Any]) -> tuple[str, str]:
    return (rec["agoge_run_id"], rec["host"]["hostname"])


def marker_sort_key(item: dict[str, Any]) -> tuple[int, datetime]:
    return (item["monotonic_ns"], parse_rfc3339_utc(item["timestamp_utc"]))


def latest_matching_marker(
    run_markers: list[dict[str, Any]], sample: dict[str, Any]
) -> dict[str, Any] | None:
    keys = [item["monotonic_ns"] for item in run_markers]
    index = bisect.bisect_right(keys, sample["monotonic_ns"]) - 1
    while index >= 0:
        marker = run_markers[index]
        if gpu_identities_match(marker.get("gpu"), sample.get("gpu")):
            return marker
        index -= 1
    return None


def join_samples(
    markers: list[dict[str, Any]], samples: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for marker in markers:
        by_key.setdefault(join_bucket_key(marker), []).append(marker)
    for run_markers in by_key.values():
        run_markers.sort(key=marker_sort_key)

    joined: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for sample in samples:
        chosen = latest_matching_marker(by_key.get(join_bucket_key(sample), []), sample)
        if chosen is not None:
            joined.append((chosen, sample))
    return joined


def check_fixture_idle_join(pairs: list[tuple[dict[str, Any], dict[str, Any]]]) -> None:
    """Fixture-only: idle 0% util still joins train step 0 from the marker."""
    idle = [
        (marker, sample)
        for marker, sample in pairs
        if sample["monotonic_ns"] == FIXTURE_IDLE_MONOTONIC_NS
    ]
    if not idle:
        return
    marker, sample = idle[0]
    util = sample["utilization_gpu"]
    require(marker["phase"] == "train", "idle fixture sample must join phase=train")
    require(util["status"] == "ok", "idle fixture sample has ok GPU util")
    require(util["value"] == 0.0, "first sample is a legitimate zero util")
    require(marker["global_step"] == 0, "idle util must still join step 0, not an inferred idle phase")


def expect_schema_error(fn: Any) -> None:
    try:
        fn()
    except SchemaError:
        return
    raise SchemaError("expected a schema error")


def slim_record(run_id: str, hostname: str, gpu: dict[str, Any], monotonic_ns: int, timestamp_utc: str) -> dict[str, Any]:
    return {
        "agoge_run_id": run_id,
        "host": {"hostname": hostname},
        "gpu": gpu,
        "monotonic_ns": monotonic_ns,
        "timestamp_utc": timestamp_utc,
    }


def check_gpu_join_guards() -> None:
    marker = slim_record("r", "h", {"uuid": "GPU-A"}, 1, "2026-01-01T00:00:00Z")
    sample = slim_record("r", "h", {"uuid": "GPU-B"}, 2, "2026-01-01T00:00:00Z")
    require(not join_samples([marker], [sample]), "mismatched GPU uuid must not join")
    pci_marker = slim_record("r", "h", {"pci_bus_id": "0000:01:00.0"}, 1, "2026-01-01T00:00:00Z")
    pci_sample = slim_record(
        "r", "h", {"uuid": "GPU-B", "pci_bus_id": "0000:01:00.0"}, 2, "2026-01-01T00:00:00Z"
    )
    require(join_samples([pci_marker], [pci_sample]), "pci fallback must join when uuids are not both set")


def check_clock_jump_refused() -> None:
    marker = slim_record("r", "h", {"uuid": "GPU-A"}, 1, "2026-01-01T00:00:10Z")
    sample = slim_record("r", "h", {"uuid": "GPU-A"}, 3, "2026-01-01T00:00:01Z")
    pairs = join_samples([marker], [sample])
    report = f0_clock.calibrate(f0_clock.observations_from_records([marker, sample]))
    ok_pairs, degraded, refused = f0_clock.annotate_joins(pairs, report["discontinuities"])
    require(report["validity"] == "refused", "backward wall jump must refuse correlation")
    require(report["join_order"] == "monotonic", "join order must stay monotonic")
    require(not ok_pairs and not degraded and refused, "join across backward jump must be refused")


def check_contract_guards() -> None:
    check_missing_is_not_zero()
    expect_schema_error(lambda: check_ok_numeric(float("nan"), "loss"))
    expect_schema_error(lambda: check_measurement({"unit": "W", "status": "unavailable"}, "power", "W"))
    expect_schema_error(lambda: parse_rfc3339_utc("not-a-dateZ"))
    expect_schema_error(lambda: check_percent_bounds({"status": "ok", "value": 150}, "utilization_gpu"))
    check_throttle({"status": "unsupported", "reasons": []})
    check_gpu_join_guards()
    check_clock_jump_refused()


def check_missing_is_not_zero() -> None:
    bad = {"value": 0, "unit": "W", "status": "unavailable"}
    try:
        check_measurement(bad, "power", "W")
    except SchemaError:
        return
    raise SchemaError("unavailable power encoded as 0 must be rejected")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--markers",
        type=Path,
        default=Path("fixtures/f0-correlation/agoge-markers.jsonl"),
    )
    parser.add_argument(
        "--samples",
        type=Path,
        default=Path("fixtures/f0-correlation/bkl-gpu-samples.jsonl"),
    )
    parser.add_argument(
        "--clock-skew",
        type=Path,
        default=Path("fixtures/f0-correlation/clock-skew.json"),
    )
    return parser.parse_args(argv[1:])


def gate_joins(
    markers: list[dict[str, Any]], samples: list[dict[str, Any]]
) -> tuple[
    dict[str, Any],
    list[tuple[dict[str, Any], dict[str, Any]]],
    list[tuple[dict[str, Any], dict[str, Any]]],
    list[tuple[dict[str, Any], dict[str, Any]]],
]:
    pairs = join_samples(markers, samples)
    report = f0_clock.calibrate(f0_clock.observations_from_records(markers + samples))
    ok_pairs, degraded, refused = f0_clock.annotate_joins(pairs, report["discontinuities"])
    return report, ok_pairs, degraded, refused


def same_optional_drift(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-6)


def check_recomputed_skew(rec: dict[str, Any], recomputed: dict[str, Any]) -> None:
    require(recomputed["validity"] == rec["validity"], "clock-skew fixture validity is stale")
    require(recomputed["offset_ns"] == rec["offset_ns"], "clock-skew fixture offset_ns is stale")
    require(recomputed["end_offset_ns"] == rec["end_offset_ns"], "clock-skew fixture end_offset_ns is stale")
    require(same_optional_drift(recomputed["drift_ns_per_s"], rec["drift_ns_per_s"]), "clock-skew fixture drift is stale")
    require(
        recomputed["correlation_across_discontinuity"] == rec["correlation_across_discontinuity"],
        "clock-skew fixture correlation policy is stale",
    )
    require(recomputed["join_order"] == rec["join_order"], "clock-skew fixture join_order is stale")


def check_derived_skew(rec: dict[str, Any], derived: dict[str, Any]) -> None:
    require(rec["validity"] == derived["validity"], "clock-skew fixture validity disagrees with stream")
    require(rec["offset_ns"] == derived["offset_ns"], "clock-skew fixture offset disagrees with stream")
    rec_kinds = [item["kind"] for item in rec["discontinuities"]]
    derived_kinds = [item["kind"] for item in derived["discontinuities"]]
    require(rec_kinds == derived_kinds, "clock-skew fixture discontinuities disagree with stream")


def check_clock_skew_fixture(path: Path, derived: dict[str, Any]) -> None:
    rec = json.loads(path.read_text())
    require(isinstance(rec, dict), f"{path}: expected object")
    f0_clock.check_clock_skew_record(rec)
    recomputed = f0_clock.calibrate(
        rec["observations"],
        assumed_utc_bound_ns=rec["assumed_utc_bound_ns"],
        clock_resolution=rec["clock_resolution"],
    )
    check_recomputed_skew(rec, recomputed)
    check_derived_skew(rec, derived)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        check_contract_guards()
        markers = load_jsonl(args.markers)
        samples = load_jsonl(args.samples)
        for rec in markers:
            check_marker(rec)
        for rec in samples:
            check_sample(rec)
        report, ok_pairs, degraded, refused = gate_joins(markers, samples)
        accepted = ok_pairs + degraded
        require(accepted, "expected at least one join on agoge_run_id + monotonic_ns")
        check_fixture_idle_join(accepted)
        check_clock_skew_fixture(args.clock_skew, report)
    except (SchemaError, f0_clock.ClockError, OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(error, file=sys.stderr)
        return 1

    print(
        f"{args.markers}: {len(markers)} markers; "
        f"{args.samples}: {len(samples)} samples; "
        f"joined_ok={len(ok_pairs)} joined_degraded={len(degraded)} refused={len(refused)}; "
        f"clock_validity={report['validity']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
