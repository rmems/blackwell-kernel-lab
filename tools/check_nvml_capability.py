#!/usr/bin/env python3
"""Validate NVML capability snapshots and run fake-backend self-tests (CPU).

See docs/f0-nvml-capability.md. Used by .github/workflows/ci-cpu.yml.

Usage:
    python3 tools/check_nvml_capability.py
    python3 tools/check_nvml_capability.py --fixture fixtures/f0-nvml-capability/full-support.json
    python3 tools/check_nvml_capability.py --live --out results/nvml-capability.json \\
        --agoge-run-id local-probe --require-name "NVIDIA GeForce RTX 5080"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

from f0_measurements import (
    MISSING_STATUSES,
    NUMERIC_METRIC_NAMES,
    SchemaError,
    THROTTLE_METRIC,
    check_measurement,
    require,
)
from nvml_capability import NVML_ERROR_DRIVER_NOT_LOADED, ProbeFailure, discover, map_nvml_error
from nvml_fakes import (
    FakeNvmlBackend,
    correlation_capability_snapshot,
    discover_scenario,
    scenario_table,
)
from nvml_live import discover_live
from nvml_schema import bind_samples, check_capability_snapshot, reject_fabricated_zero

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "fixtures" / "f0-nvml-capability"
CORRELATION_CAPABILITY = REPO_ROOT / "fixtures" / "f0-correlation" / "bkl-gpu-capability.json"
CORRELATION_SAMPLES = REPO_ROOT / "fixtures" / "f0-correlation" / "bkl-gpu-samples.jsonl"
SCENARIO_NAMES = (
    "full-support",
    "partial-support",
    "permission-denied",
    "transient-failure",
    "device-lost",
    "no-device",
    "valid-zero",
)


def load_json(path: Path) -> Any:
    text = path.read_text()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as error:
        raise SchemaError(f"{path}: {error}") from error
    return obj


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
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


def dump_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def canonical(obj: dict[str, Any]) -> str:
    return json.dumps(obj, indent=2, sort_keys=True) + "\n"


def expect_schema_error(fn: Callable[[], None], message: str) -> None:
    try:
        fn()
    except SchemaError:
        return
    raise SchemaError(message)


def metric_status(snapshot: dict[str, Any], name: str) -> str:
    return snapshot["metrics"][name]["status"]


def metric_value(snapshot: dict[str, Any], name: str) -> Any:
    return snapshot["metrics"][name]["value"]


def assert_complete_metrics(snapshot: dict[str, Any]) -> None:
    metrics = snapshot["metrics"]
    for name in NUMERIC_METRIC_NAMES:
        require(name in metrics, f"missing metric {name} after partial failure")
    require(THROTTLE_METRIC in metrics, "missing throttle after partial failure")
    require("performance_state" in metrics, "missing performance_state after partial failure")


def check_full_support() -> dict[str, Any]:
    snapshot = discover_scenario("full-support")
    check_capability_snapshot(snapshot)
    require(snapshot["device_status"] == "ok", "full-support device_status")
    require(snapshot["gpu"]["uuid"] == "GPU-fixture-0001", "full-support uuid")
    require(snapshot["gpu"]["pci_bus_id"] == "0000:01:00.0", "full-support pci")
    require(snapshot["tools"]["nvml_version"] == "13.3.58", "full-support nvml version")
    require(snapshot["tools"]["driver_version"] == "610.43.03", "full-support driver")
    require(metric_status(snapshot, "power") == "ok", "full-support power")
    require(metric_value(snapshot, "power") == 180.0, "full-support power value")
    return snapshot


def check_partial_support() -> dict[str, Any]:
    snapshot = discover_scenario("partial-support")
    check_capability_snapshot(snapshot)
    require(snapshot["device_status"] == "ok", "partial-support continues after unsupported metric")
    require(metric_status(snapshot, "temperature_memory") == "unsupported", "partial temp mem")
    require(metric_value(snapshot, "temperature_memory") is None, "partial temp mem null")
    require(metric_status(snapshot, "power") == "ok", "partial power still sampled")
    assert_complete_metrics(snapshot)
    return snapshot


def check_permission_denied() -> dict[str, Any]:
    snapshot = discover_scenario("permission-denied")
    check_capability_snapshot(snapshot)
    require(metric_status(snapshot, "power") == "permission_denied", "permission power")
    require(metric_value(snapshot, "power") is None, "permission power must not be zero")
    require(metric_status(snapshot, "utilization_gpu") == "ok", "permission util continues")
    require(snapshot["gpu"]["uuid"], "permission still records UUID")
    assert_complete_metrics(snapshot)
    return snapshot


def check_transient_failure() -> dict[str, Any]:
    snapshot = discover_scenario("transient-failure")
    check_capability_snapshot(snapshot)
    require(metric_status(snapshot, "power") == "transient_failure", "transient power")
    require(metric_value(snapshot, "power") is None, "transient power null")
    require(metric_status(snapshot, "clock_graphics") == "unavailable", "transient clock")
    require(metric_status(snapshot, "utilization_gpu") == "ok", "transient util continues")
    assert_complete_metrics(snapshot)
    return snapshot


def check_device_lost() -> dict[str, Any]:
    snapshot = discover_scenario("device-lost")
    check_capability_snapshot(snapshot)
    require(snapshot["device_status"] == "device_lost", "device-lost status")
    require(snapshot["gpu"]["uuid"] == "GPU-fixture-0001", "device-lost keeps UUID")
    require(metric_status(snapshot, "power") == "device_lost", "device-lost power")
    require(metric_value(snapshot, "power") is None, "device-lost power null")
    assert_complete_metrics(snapshot)
    return snapshot


def check_no_device() -> dict[str, Any]:
    snapshot = discover_scenario("no-device")
    check_capability_snapshot(snapshot)
    require(snapshot["device_status"] == "no_device", "no-device status")
    require(snapshot["gpu"]["uuid"] is None, "no-device uuid null")
    require(snapshot["gpu"]["pci_bus_id"] is None, "no-device pci null")
    require(metric_status(snapshot, "power") == "unsupported", "no-device power")
    require(metric_value(snapshot, "power") is None, "no-device power null")
    return snapshot


def check_valid_zero() -> dict[str, Any]:
    snapshot = discover_scenario("valid-zero")
    check_capability_snapshot(snapshot)
    util = snapshot["metrics"]["utilization_gpu"]
    require(util["status"] == "ok", "valid-zero util status")
    require(util["value"] == 0.0, "valid-zero is a legitimate 0%")
    require(snapshot["metrics"]["power"]["value"] == 15.0, "idle power is not fabricated 0")
    return snapshot


def check_zero_substitution_guards() -> None:
    check_measurement({"value": 0.0, "unit": "%", "status": "ok"}, "utilization_gpu", "%")
    for status in sorted(MISSING_STATUSES):
        reject_fabricated_zero(status)
    expect_schema_error(
        lambda: check_measurement({"value": None, "unit": "W", "status": "ok"}, "power", "W"),
        "ok power with null value must be rejected",
    )


def check_malformed_timestamp() -> None:
    snapshot = discover_scenario("full-support")
    snapshot["timestamp_utc"] = "not-a-dateZ"
    expect_schema_error(
        lambda: check_capability_snapshot(snapshot),
        "malformed timestamp must be rejected",
    )


def check_driver_not_loaded_is_unsupported() -> None:
    require(
        map_nvml_error(NVML_ERROR_DRIVER_NOT_LOADED) == "unsupported",
        "DRIVER_NOT_LOADED must not be classified as device_lost",
    )


def check_bind_rejects_other_gpu() -> None:
    snapshot = correlation_capability_snapshot()
    sample = json.loads(json.dumps(load_jsonl(CORRELATION_SAMPLES)[0]))
    sample["gpu"]["uuid"] = "GPU-other"
    expect_schema_error(
        lambda: bind_samples(snapshot, [sample]),
        "sample from another GPU must not bind",
    )


def check_missing_gpu_key_rejected() -> None:
    snapshot = discover_scenario("full-support")
    del snapshot["gpu"]["name"]
    expect_schema_error(
        lambda: check_capability_snapshot(snapshot),
        "omitted gpu.name must be rejected",
    )


def check_optional_failure_does_not_abort() -> None:
    class BoomPower(FakeNvmlBackend):
        def read_numeric(self, index: int, metric: str) -> int | float:
            if metric == "power":
                raise ProbeFailure("transient_failure", "boom")
            return super().read_numeric(index, metric)

    snapshot = discover(
        BoomPower("full-support"),
        agoge_run_id="cap_optional_failure",
        hostname="ShipOfTheseus",
        timestamp_utc="2026-09-15T04:00:00Z",
        monotonic_ns=0,
    )
    check_capability_snapshot(snapshot)
    require(metric_status(snapshot, "power") == "transient_failure", "boom power")
    require(metric_status(snapshot, "temperature_gpu") == "ok", "other metrics survive")
    require(metric_status(snapshot, "vram_total") == "ok", "vram survives")


def check_committed_fixtures(snapshots: dict[str, dict[str, Any]]) -> None:
    for name, snapshot in snapshots.items():
        path = FIXTURE_DIR / f"{name}.json"
        require(path.is_file(), f"missing fixture {path}")
        loaded = load_json(path)
        require(canonical(loaded) == canonical(snapshot), f"{path} drifted from fake backend")
        check_capability_snapshot(loaded)
    bound = load_json(CORRELATION_CAPABILITY)
    check_capability_snapshot(bound)
    expected = correlation_capability_snapshot()
    require(canonical(bound) == canonical(expected), f"{CORRELATION_CAPABILITY} drifted")
    bind_samples(bound, load_jsonl(CORRELATION_SAMPLES))


def run_self_test() -> None:
    check_zero_substitution_guards()
    check_malformed_timestamp()
    check_driver_not_loaded_is_unsupported()
    check_bind_rejects_other_gpu()
    check_missing_gpu_key_rejected()
    snapshots = {
        "full-support": check_full_support(),
        "partial-support": check_partial_support(),
        "permission-denied": check_permission_denied(),
        "transient-failure": check_transient_failure(),
        "device-lost": check_device_lost(),
        "no-device": check_no_device(),
        "valid-zero": check_valid_zero(),
    }
    require(set(snapshots) == set(scenario_table()), "scenario table mismatch")
    check_optional_failure_does_not_abort()
    check_committed_fixtures(snapshots)


def write_fixtures() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    for name in SCENARIO_NAMES:
        dump_json(FIXTURE_DIR / f"{name}.json", discover_scenario(name))
    dump_json(CORRELATION_CAPABILITY, correlation_capability_snapshot())


def stamp_sample_digests() -> None:
    digest = correlation_capability_snapshot()["capability_digest"]
    samples = load_jsonl(CORRELATION_SAMPLES)
    lines = []
    for sample in samples:
        sample["capability_digest"] = digest
        lines.append(json.dumps(sample, separators=(",", ":")))
    CORRELATION_SAMPLES.write_text("\n".join(lines) + "\n")


def check_live_requirements(snapshot: dict[str, Any], args: argparse.Namespace) -> None:
    check_capability_snapshot(snapshot)
    if args.require_device and snapshot["device_status"] != "ok":
        raise SchemaError(f"live device_status={snapshot['device_status']} (NVML probe did not see a GPU)")
    name = snapshot["gpu"].get("name")
    if args.require_name and name != args.require_name:
        raise SchemaError(f"live GPU name {name!r} != {args.require_name!r}")
    capability = snapshot["gpu"].get("compute_capability")
    if args.require_compute_capability and capability != args.require_compute_capability:
        raise SchemaError(
            f"live compute_capability {capability!r} != {args.require_compute_capability!r}"
        )


def run_live(args: argparse.Namespace) -> dict[str, Any]:
    run_id = args.agoge_run_id if args.agoge_run_id else "live-nvml-capability"
    snapshot = discover_live(agoge_run_id=run_id)
    check_live_requirements(snapshot, args)
    if args.out is not None:
        dump_json(args.out, snapshot)
    return snapshot


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, help="Validate one capability JSON snapshot")
    parser.add_argument("--self-test", action="store_true", help="Run fake-backend tests (default)")
    parser.add_argument("--write-fixtures", action="store_true", help="Regenerate committed JSON fixtures")
    parser.add_argument("--stamp-samples", action="store_true", help="Stamp capability_digest on F0 samples")
    parser.add_argument("--live", action="store_true", help="Probe live NVML (RTX 5080 recipe)")
    parser.add_argument("--out", type=Path, help="Write snapshot JSON (live or fixture dump)")
    parser.add_argument("--agoge-run-id", help="Run id stamped on a live snapshot")
    parser.add_argument("--require-device", action="store_true", help="Fail live probe if no GPU")
    parser.add_argument("--require-name", help="Fail live probe unless gpu.name matches")
    parser.add_argument(
        "--require-compute-capability",
        help="Fail live probe unless gpu.compute_capability matches",
    )
    parser.add_argument(
        "--bind-samples",
        type=Path,
        help="JSONL samples that must carry this snapshot's capability_digest",
    )
    parser.add_argument(
        "--capability",
        type=Path,
        default=CORRELATION_CAPABILITY,
        help="Capability snapshot used with --bind-samples",
    )
    return parser.parse_args(argv[1:])


def dispatch(args: argparse.Namespace) -> str:
    if args.write_fixtures:
        write_fixtures()
        return f"wrote fixtures under {FIXTURE_DIR}"
    if args.stamp_samples:
        stamp_sample_digests()
        return f"stamped {CORRELATION_SAMPLES}"
    if args.fixture is not None:
        snapshot = load_json(args.fixture)
        check_capability_snapshot(snapshot)
        return f"{args.fixture}: capability snapshot ok digest={snapshot['capability_digest']}"
    if args.live:
        snapshot = run_live(args)
        return (
            f"live device_status={snapshot['device_status']} "
            f"name={snapshot['gpu'].get('name')!r} digest={snapshot['capability_digest']}"
        )
    if args.bind_samples is not None:
        bind_samples(load_json(args.capability), load_jsonl(args.bind_samples))
        return f"bound {args.bind_samples} to {args.capability}"
    run_self_test()
    return (
        f"nvml capability self-test ok; scenarios={len(SCENARIO_NAMES)}; "
        f"bound {CORRELATION_SAMPLES.name}"
    )


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        print(dispatch(args))
    except (SchemaError, OSError, KeyError, TypeError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
