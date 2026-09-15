#!/usr/bin/env python3
"""Validate F0 one-host clock-skew fixtures (CPU, no GPU).

See docs/f0-correlation-schema.md. Used by .github/workflows/ci-cpu.yml.

Usage:
    python3 tools/check_f0_clock_skew.py
    python3 tools/check_f0_clock_skew.py --fixtures fixtures/f0-correlation/clock-skew
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import f0_clock
from check_f0_correlation import join_samples

SCENARIO_NAMES = (
    "stable",
    "bounded_drift",
    "backward_jump",
    "forward_jump",
    "sparse_markers",
    "missing_end",
)


def require(condition: object, message: str) -> None:
    if not condition:
        raise f0_clock.ClockError(message)


def load_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text())
    require(isinstance(obj, dict), f"{path}: expected object")
    return obj


def scenario_paths(root: Path) -> list[Path]:
    paths = []
    for name in SCENARIO_NAMES:
        path = root / f"{name}.json"
        require(path.is_file(), f"missing scenario fixture {path}")
        paths.append(path)
    return paths


def expected_kinds(expect: dict[str, Any]) -> list[str]:
    kinds = expect.get("discontinuity_kinds")
    require(isinstance(kinds, list), "expect.discontinuity_kinds must be a list")
    for kind in kinds:
        f0_clock.check_discontinuity_kind(kind)
    return list(kinds)


def check_expect_validity(report: dict[str, Any], expect: dict[str, Any]) -> None:
    expected = expect.get("validity")
    f0_clock.check_validity(expected)
    require(report["validity"] == expected, f"validity {report['validity']!r} != {expected!r}")
    expected_cross = expect.get("correlation_across_discontinuity")
    if expected_cross is None:
        return
    f0_clock.check_cross_policy(expected_cross)
    require(
        report["correlation_across_discontinuity"] == expected_cross,
        "correlation_across_discontinuity mismatch",
    )


def check_expect_kinds(report: dict[str, Any], expect: dict[str, Any]) -> None:
    got = [item["kind"] for item in report["discontinuities"]]
    expected = expected_kinds(expect)
    require(got == expected, f"discontinuity kinds {got} != {expected}")


def check_expect_drift(report: dict[str, Any], expect: dict[str, Any]) -> None:
    if "drift_ns_per_s" not in expect:
        return
    expected = expect["drift_ns_per_s"]
    got = report["drift_ns_per_s"]
    if expected is None:
        require(got is None, f"expected null drift, got {got!r}")
        return
    require(got is not None, "expected numeric drift, got null")
    require(math.isclose(got, expected, rel_tol=0.0, abs_tol=1e-6), f"drift {got} != {expected}")


def check_expect_offset(report: dict[str, Any], expect: dict[str, Any]) -> None:
    if not expect.get("offset_matches_start", True):
        return
    start = report["start_observation"]
    require(report["offset_ns"] == f0_clock.offset_ns_of(start), "offset_ns must match start pair")


def check_expect_end(report: dict[str, Any], expect: dict[str, Any]) -> None:
    has_end = expect.get("has_end_observation")
    if has_end is None:
        return
    require(isinstance(has_end, bool), "expect.has_end_observation must be a bool")
    require((report["end_observation"] is not None) == has_end, "end observation presence mismatch")


def join_counts(
    scenario: dict[str, Any], discontinuities: list[dict[str, Any]]
) -> tuple[int, int, int]:
    markers = scenario.get("markers")
    samples = scenario.get("samples")
    if not markers and not samples:
        return (0, 0, 0)
    require(isinstance(markers, list) and isinstance(samples, list), "markers/samples must be lists")
    pairs = join_samples(markers, samples)
    ok_pairs, degraded, refused = f0_clock.annotate_joins(pairs, discontinuities)
    return len(ok_pairs), len(degraded), len(refused)


def check_expect_joins(scenario: dict[str, Any], report: dict[str, Any]) -> None:
    expect = scenario["expect"]
    ok_n, degraded_n, refused_n = join_counts(scenario, report["discontinuities"])
    require(ok_n == expect["joined_ok"], f"joined_ok {ok_n} != {expect['joined_ok']}")
    require(
        degraded_n == expect["joined_degraded"],
        f"joined_degraded {degraded_n} != {expect['joined_degraded']}",
    )
    require(refused_n == expect["refused"], f"refused {refused_n} != {expect['refused']}")


def calibrate_scenario(scenario: dict[str, Any]) -> dict[str, Any]:
    resolution = scenario.get("clock_resolution")
    bound = scenario.get("assumed_utc_bound_ns", f0_clock.UTC_BOUND_NS)
    return f0_clock.calibrate(
        scenario["observations"],
        assumed_utc_bound_ns=bound,
        clock_resolution=resolution,
    )


def check_record_roundtrip(report: dict[str, Any], scenario: dict[str, Any]) -> None:
    envelope = scenario.get("envelope")
    if not isinstance(envelope, dict):
        return
    record = f0_clock.build_clock_skew_record(
        report,
        agoge_run_id=envelope["agoge_run_id"],
        host=envelope["host"],
        gpu=envelope["gpu"],
        collector=envelope["collector"],
    )
    f0_clock.check_clock_skew_record(record)


def run_scenario(path: Path) -> None:
    scenario = load_json(path)
    require(scenario.get("name") == path.stem, f"{path}: name must match filename stem")
    require(isinstance(scenario.get("expect"), dict), f"{path}: expect object required")
    report = calibrate_scenario(scenario)
    f0_clock.check_clock_skew_payload(report)
    check_expect_validity(report, scenario["expect"])
    check_expect_kinds(report, scenario["expect"])
    check_expect_drift(report, scenario["expect"])
    check_expect_offset(report, scenario["expect"])
    check_expect_end(report, scenario["expect"])
    check_expect_joins(scenario, report)
    check_record_roundtrip(report, scenario)


def check_capture_shape() -> None:
    obs = f0_clock.capture_observation("start")
    f0_clock.normalize_one(obs)
    require(obs["role"] == "start", "capture_observation must stamp role=start")
    require(obs["monotonic_ns"] >= 0, "captured monotonic_ns must be non-negative")


def check_unknown_kind_rejected() -> None:
    try:
        f0_clock.check_discontinuity_kind("ntp_vibes")
    except f0_clock.ClockError:
        return
    raise f0_clock.ClockError("unknown discontinuity kind must be rejected")


def check_missing_end_drift_is_null() -> None:
    report = f0_clock.calibrate(
        [{"role": "start", "timestamp_utc": "2026-09-14T18:00:00Z", "monotonic_ns": 1_000}]
    )
    require(report["validity"] == "degraded", "missing end must be degraded")
    require(report["drift_ns_per_s"] is None, "missing end must not invent drift")
    require(report["end_observation"] is None, "missing end observation must stay null")


def check_monotonic_regression() -> None:
    report = f0_clock.calibrate(
        [
            {"role": "start", "timestamp_utc": "2026-09-14T18:00:00Z", "monotonic_ns": 2000},
            {"role": "end", "timestamp_utc": "2026-09-14T18:00:01Z", "monotonic_ns": 1000},
        ]
    )
    kinds = [item["kind"] for item in report["discontinuities"]]
    require("monotonic_regression" in kinds, "decreasing monotonic_ns must be recorded")
    require(report["validity"] == "refused", "monotonic regression must refuse correlation")


def check_contract_guards() -> None:
    check_capture_shape()
    check_unknown_kind_rejected()
    check_missing_end_drift_is_null()
    check_monotonic_regression()
    f0_clock.check_validity("ok")
    f0_clock.check_validity("degraded")
    f0_clock.check_validity("refused")
    resolution = f0_clock.read_clock_resolution()
    require(resolution["monotonic_ns"] >= 1, "monotonic resolution must be at least 1 ns")
    require(resolution["realtime_ns"] >= 1, "realtime resolution must be at least 1 ns")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=Path("fixtures/f0-correlation/clock-skew"),
    )
    return parser.parse_args(argv[1:])


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        check_contract_guards()
        paths = scenario_paths(args.fixtures)
        for path in paths:
            run_scenario(path)
    except (f0_clock.ClockError, OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(error, file=sys.stderr)
        return 1

    names = ", ".join(path.stem for path in paths)
    print(f"{args.fixtures}: {len(paths)} scenarios ok ({names})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
