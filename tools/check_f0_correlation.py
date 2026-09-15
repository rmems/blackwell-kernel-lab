#!/usr/bin/env python3
"""Validate and join F0 Agoge ↔ BKL correlation fixtures (CPU, no GPU).

See docs/f0-correlation-schema.md. Used by .github/workflows/ci-cpu.yml.

Usage:
    python3 tools/check_f0_correlation.py \\
      --markers fixtures/f0-correlation/agoge-markers.jsonl \\
      --samples fixtures/f0-correlation/bkl-gpu-samples.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCHEMA = "bkl.f0_correlation.v1"
MARKER = "agoge_marker"
SAMPLE = "bkl_gpu_sample"
MEASUREMENT_STATUS = {"ok", "unavailable", "unsupported"}

PHYSICAL = (
    ("power", "W"),
    ("temperature_gpu", "C"),
    ("temperature_memory", "C"),
    ("utilization_gpu", "%"),
    ("utilization_memory", "%"),
    ("clock_graphics", "MHz"),
    ("clock_memory", "MHz"),
    ("vram_used", "MiB"),
    ("vram_free", "MiB"),
    ("vram_total", "MiB"),
    ("headroom", "MiB"),
)


class SchemaError(Exception):
    """A record or join failed validation."""


def require(condition: object, message: str) -> None:
    if not condition:
        raise SchemaError(message)


def require_nonempty_str(value: Any, message: str) -> None:
    require(isinstance(value, str), message)
    require(value, message)


def require_optional_nonempty_str(value: Any, message: str) -> None:
    if value is None:
        return
    require_nonempty_str(value, message)


def require_nonneg_int(value: Any, message: str) -> None:
    require(isinstance(value, int), message)
    require(not isinstance(value, bool), message)
    require(value >= 0, message)


def require_optional_nonneg_int(value: Any, message: str) -> None:
    if value is None:
        return
    require_nonneg_int(value, message)


def require_positive_int(value: Any, message: str) -> None:
    require_nonneg_int(value, message)
    require(value > 0, message)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    text = path.read_text()
    for line_no, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError as error:
            raise SchemaError(f"{path}:{line_no}: {error}") from error
        require(isinstance(obj, dict), f"{path}:{line_no}: expected object")
        rows.append(obj)
    require(rows, f"{path}: no records")
    return rows


def check_measurement(obj: Any, name: str, unit: str) -> None:
    require(isinstance(obj, dict), f"{name} must be a measurement object")
    require(obj.get("unit") == unit, f"{name}: expected unit {unit}, got {obj.get('unit')}")
    status = obj.get("status")
    require(status in MEASUREMENT_STATUS, f"{name}: bad status {status}")
    value = obj.get("value")
    if status == "ok":
        require(isinstance(value, (int, float)), f"{name}: ok requires a numeric value, not {value!r}")
        require(not isinstance(value, bool), f"{name}: ok requires a numeric value, not {value!r}")
        return
    require(value is None, f"{name}: missing must be null, not {value!r} (missing ≠ zero)")


def check_host(rec: dict[str, Any]) -> None:
    host = rec.get("host")
    require(isinstance(host, dict), "host object required")
    require_nonempty_str(host.get("hostname"), "host.hostname required")


def check_time(rec: dict[str, Any]) -> None:
    timestamp = rec.get("timestamp_utc")
    require(isinstance(timestamp, str), "timestamp_utc must be UTC RFC3339 ending in Z")
    require(timestamp.endswith("Z"), "timestamp_utc must be UTC RFC3339 ending in Z")
    require_nonneg_int(rec.get("monotonic_ns"), "monotonic_ns must be a non-negative int")


def check_collector(rec: dict[str, Any]) -> None:
    collector = rec.get("collector")
    require(isinstance(collector, dict), "collector object required")
    require(collector.get("id"), "collector.id and collector.version required")
    require(collector.get("version"), "collector.id and collector.version required")


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


def check_throttle(obj: Any) -> None:
    require(isinstance(obj, dict), "throttle must be an object")
    status = obj.get("status")
    require(status in MEASUREMENT_STATUS, f"throttle: bad status {status}")
    reasons = obj.get("reasons")
    if status == "ok":
        require(isinstance(reasons, list), "throttle.reasons must be a list when ok")
        require(all(isinstance(item, str) for item in reasons), "throttle.reasons must be strings")
        return
    require(reasons is None, "throttle.reasons must be null when not ok")


def check_sample_physical(rec: dict[str, Any]) -> None:
    for name, unit in PHYSICAL:
        require(name in rec, f"missing physical field {name}")
        check_measurement(rec[name], name, unit)


def check_cuda(rec: dict[str, Any]) -> None:
    cuda = rec.get("cuda")
    require(isinstance(cuda, dict), "cuda object required")
    require_optional_nonempty_str(cuda.get("driver_version"), "cuda.driver_version must be null or string")
    require_optional_nonempty_str(cuda.get("runtime_version"), "cuda.runtime_version must be null or string")
    require_optional_nonempty_str(cuda.get("tool"), "cuda.tool must be null or string")


def check_sample(rec: dict[str, Any]) -> None:
    check_envelope(rec, SAMPLE)
    require_positive_int(rec.get("cadence_ms"), "bkl_gpu_sample cadence_ms must be a positive int")
    check_sample_physical(rec)
    check_throttle(rec.get("throttle"))
    require_optional_nonempty_str(
        rec.get("profile_window_ref"),
        "profile_window_ref must be null or a non-empty string",
    )
    check_cuda(rec)


def join_samples(
    markers: list[dict[str, Any]], samples: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    by_run: dict[str, list[dict[str, Any]]] = {}
    for marker in markers:
        by_run.setdefault(marker["agoge_run_id"], []).append(marker)
    for run_markers in by_run.values():
        run_markers.sort(key=lambda item: (item["monotonic_ns"], item["timestamp_utc"]))

    joined: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for sample in samples:
        run_markers = by_run.get(sample["agoge_run_id"], [])
        host = sample["host"]["hostname"]
        chosen = None
        for marker in run_markers:
            if marker["host"]["hostname"] != host:
                continue
            if marker["monotonic_ns"] <= sample["monotonic_ns"]:
                chosen = marker
        if chosen is not None:
            joined.append((chosen, sample))
    return joined


def phase_from_markers_only(pairs: list[tuple[dict[str, Any], dict[str, Any]]]) -> None:
    """Joined phase must come from the Agoge marker, never from GPU utilization."""
    for marker, sample in pairs:
        util = sample["utilization_gpu"]
        require(marker["phase"] == "train", "fixture markers use phase=train")
        require(util["status"] == "ok", "fixture samples have ok GPU util")
        # First fixture sample is idle (0%) but still joins to train step 0.
        if sample["monotonic_ns"] == 1500000000:
            require(util["value"] == 0.0, "first sample is a legitimate zero util")
            require(marker["global_step"] == 0, "idle util must still join step 0, not an inferred idle phase")


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
    return parser.parse_args(argv[1:])


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        check_missing_is_not_zero()
        markers = load_jsonl(args.markers)
        samples = load_jsonl(args.samples)
        for rec in markers:
            check_marker(rec)
        for rec in samples:
            check_sample(rec)
        pairs = join_samples(markers, samples)
        require(pairs, "expected at least one join on agoge_run_id + monotonic_ns")
        phase_from_markers_only(pairs)
    except (SchemaError, OSError, KeyError, TypeError) as error:
        print(error, file=sys.stderr)
        return 1

    print(
        f"{args.markers}: {len(markers)} markers; "
        f"{args.samples}: {len(samples)} samples; "
        f"joined={len(pairs)} ok"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
