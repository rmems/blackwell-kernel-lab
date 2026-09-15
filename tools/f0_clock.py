#!/usr/bin/env python3
"""One-host wall-clock ↔ monotonic calibration for F0 correlation.

See docs/f0-correlation-schema.md. CPU-only: this does not sample the RTX 5080
and does not occupy VRAM. Pairing is for a single host/boot; there is no NTP
daemon or cross-machine sync here.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

SCHEMA = "bkl.f0_correlation.v1"
CLOCK_SKEW = "bkl_clock_skew"
JOIN_ORDER = "monotonic"
UTC_BOUND_NS = 1_000_000_000
NS_PER_S = 1_000_000_000
UNIX_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


class ClockError(Exception):
    """Clock calibration or skew-report validation failed."""


def require(condition: object, message: str) -> None:
    if not condition:
        raise ClockError(message)


def require_nonempty_str(value: Any, message: str) -> None:
    require(isinstance(value, str), message)
    require(value, message)


def require_nonneg_int(value: Any, message: str) -> None:
    require(isinstance(value, int), message)
    require(not isinstance(value, bool), message)
    require(value >= 0, message)


def parse_rfc3339_utc(timestamp: Any) -> datetime:
    require(isinstance(timestamp, str), "timestamp_utc must be UTC RFC3339 ending in Z")
    require(timestamp.endswith("Z"), "timestamp_utc must be UTC RFC3339 ending in Z")
    try:
        return datetime.fromisoformat(timestamp[:-1] + "+00:00")
    except ValueError as error:
        raise ClockError("timestamp_utc must be UTC RFC3339 ending in Z") from error


def datetime_to_ns(dt: datetime) -> int:
    delta = dt.astimezone(timezone.utc) - UNIX_EPOCH
    return (delta.days * 86400 + delta.seconds) * NS_PER_S + delta.microseconds * 1000


def rfc3339_to_ns(timestamp: str) -> int:
    return datetime_to_ns(parse_rfc3339_utc(timestamp))


def ns_to_rfc3339_utc(ns: int) -> str:
    seconds, frac = divmod(int(ns), NS_PER_S)
    dt = datetime.fromtimestamp(seconds, tz=timezone.utc)
    base = dt.strftime("%Y-%m-%dT%H:%M:%S")
    if frac == 0:
        return base + "Z"
    frac_str = f"{frac:09d}".rstrip("0")
    return f"{base}.{frac_str}Z"


def res_ns(clock_id: int) -> int:
    try:
        resolution = time.clock_getres(clock_id)
    except OSError:
        return 1
    return max(1, int(resolution * NS_PER_S))


def read_clock_resolution() -> dict[str, int]:
    return {
        "monotonic_ns": res_ns(time.CLOCK_MONOTONIC),
        "realtime_ns": res_ns(time.CLOCK_REALTIME),
    }


def capture_observation(role: str) -> dict[str, Any]:
    check_role(role)
    wall_ns = time.time_ns()
    monotonic_ns = time.monotonic_ns()
    return {
        "role": role,
        "timestamp_utc": ns_to_rfc3339_utc(wall_ns),
        "monotonic_ns": monotonic_ns,
        "wall_ns": wall_ns,
    }


def check_role(role: Any) -> None:
    if role == "start":
        return
    if role == "end":
        return
    if role == "sample":
        return
    raise ClockError(f"observation role must be start, end, or sample, got {role!r}")


def check_validity(value: Any) -> None:
    if value == "ok":
        return
    if value == "degraded":
        return
    if value == "refused":
        return
    raise ClockError(f"validity must be ok, degraded, or refused, got {value!r}")


def check_cross_policy(value: Any) -> None:
    if value == "allowed":
        return
    if value == "degraded":
        return
    if value == "refused":
        return
    raise ClockError(
        "correlation_across_discontinuity must be allowed, degraded, or refused, "
        f"got {value!r}"
    )


def check_discontinuity_kind(kind: Any) -> None:
    if kind == "backward_wall_clock":
        return
    if kind == "forward_jump":
        return
    if kind == "monotonic_regression":
        return
    raise ClockError(f"unknown discontinuity kind {kind!r}")


def discontinuity_policy(kind: str) -> str:
    check_discontinuity_kind(kind)
    if kind == "forward_jump":
        return "degraded"
    return "refused"


def worse_validity(left: str, right: str) -> str:
    check_validity(left)
    check_validity(right)
    rank = {"ok": 0, "degraded": 1, "refused": 2}
    if rank[left] >= rank[right]:
        return left
    return right


def classify_step(mono_delta: int, wall_delta: int, bound_ns: int) -> str | None:
    if mono_delta < 0:
        return "monotonic_regression"
    if wall_delta < 0:
        return "backward_wall_clock"
    offset_delta = wall_delta - mono_delta
    if offset_delta > bound_ns:
        return "forward_jump"
    if offset_delta < -bound_ns:
        return "backward_wall_clock"
    return None


def make_discontinuity(kind: str, prev: dict[str, Any], curr: dict[str, Any]) -> dict[str, Any]:
    check_discontinuity_kind(kind)
    return {
        "kind": kind,
        "from_monotonic_ns": prev["monotonic_ns"],
        "to_monotonic_ns": curr["monotonic_ns"],
        "from_timestamp_utc": prev["timestamp_utc"],
        "to_timestamp_utc": curr["timestamp_utc"],
        "monotonic_delta_ns": curr["monotonic_ns"] - prev["monotonic_ns"],
        "wall_delta_ns": curr["wall_ns"] - prev["wall_ns"],
        "correlation": discontinuity_policy(kind),
    }


def detect_monotonic_regressions(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for prev, curr in zip(observations, observations[1:]):
        if curr["monotonic_ns"] < prev["monotonic_ns"]:
            found.append(make_discontinuity("monotonic_regression", prev, curr))
    return found


def detect_wall_jumps(
    observations: list[dict[str, Any]], bound_ns: int
) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for prev, curr in zip(observations, observations[1:]):
        kind = classify_step(
            curr["monotonic_ns"] - prev["monotonic_ns"],
            curr["wall_ns"] - prev["wall_ns"],
            bound_ns,
        )
        if kind is None:
            continue
        if kind == "monotonic_regression":
            continue
        found.append(make_discontinuity(kind, prev, curr))
    return found


def has_named_role(observations: list[dict[str, Any]]) -> bool:
    for item in observations:
        if item.get("role") is not None:
            return True
    return False


def detect_discontinuities(
    observations: list[dict[str, Any]], bound_ns: int
) -> list[dict[str, Any]]:
    ordered = sorted(observations, key=lambda item: (item["monotonic_ns"], item["wall_ns"]))
    wall_jumps = detect_wall_jumps(ordered, bound_ns)
    if not has_named_role(observations):
        return wall_jumps
    return detect_monotonic_regressions(observations) + wall_jumps


def first_role(observations: list[dict[str, Any]], role: str) -> dict[str, Any] | None:
    for item in observations:
        if item.get("role") == role:
            return item
    return None


def last_role(observations: list[dict[str, Any]], role: str) -> dict[str, Any] | None:
    found = None
    for item in observations:
        if item.get("role") == role:
            found = item
    return found


def pick_start(
    observations: list[dict[str, Any]], ordered: list[dict[str, Any]]
) -> dict[str, Any]:
    explicit = first_role(observations, "start")
    if explicit is not None:
        return explicit
    return ordered[0]


def pick_end(
    observations: list[dict[str, Any]], ordered: list[dict[str, Any]]
) -> dict[str, Any] | None:
    explicit_end = last_role(observations, "end")
    if explicit_end is not None:
        return explicit_end
    if first_role(observations, "start") is not None:
        return None
    if len(ordered) >= 2:
        return ordered[-1]
    return None


def offset_ns_of(observation: dict[str, Any]) -> int:
    return observation["wall_ns"] - observation["monotonic_ns"]


def drift_ns_per_s(start: dict[str, Any], end: dict[str, Any] | None) -> float | None:
    if end is None:
        return None
    elapsed_mono = end["monotonic_ns"] - start["monotonic_ns"]
    if elapsed_mono <= 0:
        return None
    elapsed_wall = end["wall_ns"] - start["wall_ns"]
    return (elapsed_wall - elapsed_mono) * NS_PER_S / elapsed_mono


def decide_validity(*, has_end: bool, kinds: list[str]) -> str:
    status = "ok" if has_end else "degraded"
    for kind in kinds:
        status = worse_validity(status, discontinuity_policy(kind))
    return status


def correlation_across_discontinuity(kinds: list[str]) -> str:
    status = "ok"
    for kind in kinds:
        status = worse_validity(status, discontinuity_policy(kind))
    if status == "ok":
        return "allowed"
    return status


def default_resolution(clock_resolution: dict[str, int] | None) -> dict[str, int]:
    if clock_resolution is None:
        return {"monotonic_ns": 1, "realtime_ns": 1}
    require_nonneg_int(clock_resolution.get("monotonic_ns"), "clock_resolution.monotonic_ns")
    require_nonneg_int(clock_resolution.get("realtime_ns"), "clock_resolution.realtime_ns")
    require(clock_resolution["monotonic_ns"] > 0, "clock_resolution.monotonic_ns must be > 0")
    require(clock_resolution["realtime_ns"] > 0, "clock_resolution.realtime_ns must be > 0")
    return {
        "monotonic_ns": clock_resolution["monotonic_ns"],
        "realtime_ns": clock_resolution["realtime_ns"],
    }


def normalize_one(item: Any) -> dict[str, Any]:
    require(isinstance(item, dict), "clock observation must be an object")
    timestamp = item.get("timestamp_utc")
    dt = parse_rfc3339_utc(timestamp)
    monotonic_ns = item.get("monotonic_ns")
    require_nonneg_int(monotonic_ns, "monotonic_ns must be a non-negative int")
    derived_wall = datetime_to_ns(dt)
    wall_ns = item.get("wall_ns")
    if wall_ns is None:
        wall_ns = derived_wall
    else:
        require_nonneg_int(wall_ns, "wall_ns must be a non-negative int")
    role = item.get("role")
    if role is not None:
        check_role(role)
    out: dict[str, Any] = {
        "timestamp_utc": timestamp,
        "monotonic_ns": monotonic_ns,
        "wall_ns": wall_ns,
    }
    if role is not None:
        out["role"] = role
    return out


def normalize_observations(raw: list[Any]) -> list[dict[str, Any]]:
    require(isinstance(raw, list) and raw, "at least one clock observation is required")
    return [normalize_one(item) for item in raw]


def observations_from_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "timestamp_utc": rec["timestamp_utc"],
            "monotonic_ns": rec["monotonic_ns"],
        }
        for rec in records
    ]


def calibrate(
    observations: list[dict[str, Any]],
    *,
    assumed_utc_bound_ns: int = UTC_BOUND_NS,
    clock_resolution: dict[str, int] | None = None,
) -> dict[str, Any]:
    require_nonneg_int(assumed_utc_bound_ns, "assumed_utc_bound_ns must be a non-negative int")
    require(assumed_utc_bound_ns > 0, "assumed_utc_bound_ns must be > 0")
    normalized = normalize_observations(observations)
    ordered = sorted(normalized, key=lambda item: (item["monotonic_ns"], item["wall_ns"]))
    start = pick_start(normalized, ordered)
    end = pick_end(normalized, ordered)
    discontinuities = detect_discontinuities(normalized, assumed_utc_bound_ns)
    kinds = [item["kind"] for item in discontinuities]
    has_end = end is not None
    validity = decide_validity(has_end=has_end, kinds=kinds)
    return {
        "observations": normalized,
        "start_observation": start,
        "end_observation": end,
        "offset_ns": offset_ns_of(start),
        "end_offset_ns": None if end is None else offset_ns_of(end),
        "drift_ns_per_s": drift_ns_per_s(start, end),
        "clock_resolution": default_resolution(clock_resolution),
        "assumed_utc_bound_ns": assumed_utc_bound_ns,
        "discontinuities": discontinuities,
        "validity": validity,
        "join_order": JOIN_ORDER,
        "correlation_across_discontinuity": correlation_across_discontinuity(kinds),
    }


def pair_straddles(
    marker_mono: int, sample_mono: int, disc: dict[str, Any]
) -> bool:
    return marker_mono <= disc["from_monotonic_ns"] and sample_mono >= disc["to_monotonic_ns"]


def pair_join_quality(
    marker: dict[str, Any], sample: dict[str, Any], discontinuities: list[dict[str, Any]]
) -> str | None:
    quality = "ok"
    for disc in discontinuities:
        if not pair_straddles(marker["monotonic_ns"], sample["monotonic_ns"], disc):
            continue
        policy = discontinuity_policy(disc["kind"])
        if policy == "refused":
            return None
        quality = worse_validity(quality, policy)
    return quality


def annotate_joins(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    discontinuities: list[dict[str, Any]],
) -> tuple[
    list[tuple[dict[str, Any], dict[str, Any]]],
    list[tuple[dict[str, Any], dict[str, Any]]],
    list[tuple[dict[str, Any], dict[str, Any]]],
]:
    ok_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    degraded: list[tuple[dict[str, Any], dict[str, Any]]] = []
    refused: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for marker, sample in pairs:
        quality = pair_join_quality(marker, sample, discontinuities)
        if quality is None:
            refused.append((marker, sample))
            continue
        if quality == "ok":
            ok_pairs.append((marker, sample))
            continue
        if quality == "degraded":
            degraded.append((marker, sample))
            continue
        raise ClockError(f"unexpected join quality {quality!r}")
    return ok_pairs, degraded, refused


def check_clock_resolution(obj: Any) -> None:
    require(isinstance(obj, dict), "clock_resolution must be an object")
    default_resolution(obj)


def check_observation_payload(item: Any) -> None:
    normalize_one(item)


def check_discontinuity_record(item: Any) -> None:
    require(isinstance(item, dict), "discontinuity must be an object")
    check_discontinuity_kind(item.get("kind"))
    require_nonneg_int(item.get("from_monotonic_ns"), "discontinuity.from_monotonic_ns")
    require_nonneg_int(item.get("to_monotonic_ns"), "discontinuity.to_monotonic_ns")
    parse_rfc3339_utc(item.get("from_timestamp_utc"))
    parse_rfc3339_utc(item.get("to_timestamp_utc"))
    require(isinstance(item.get("monotonic_delta_ns"), int), "monotonic_delta_ns must be int")
    require(not isinstance(item.get("monotonic_delta_ns"), bool), "monotonic_delta_ns must be int")
    require(isinstance(item.get("wall_delta_ns"), int), "wall_delta_ns must be int")
    require(not isinstance(item.get("wall_delta_ns"), bool), "wall_delta_ns must be int")
    policy = item.get("correlation")
    require(policy == discontinuity_policy(item["kind"]), "discontinuity.correlation must match kind")


def check_optional_end(end: Any) -> None:
    if end is None:
        return
    check_observation_payload(end)


def check_drift(value: Any) -> None:
    if value is None:
        return
    require(isinstance(value, (int, float)), "drift_ns_per_s must be a number or null")
    require(not isinstance(value, bool), "drift_ns_per_s must be a number or null")


def check_join_order(value: Any) -> None:
    require(value == JOIN_ORDER, f"join_order must be {JOIN_ORDER!r}")


def check_clock_skew_payload(rec: dict[str, Any]) -> None:
    observations = rec.get("observations")
    require(isinstance(observations, list) and observations, "observations required")
    for item in observations:
        check_observation_payload(item)
    check_observation_payload(rec.get("start_observation"))
    check_optional_end(rec.get("end_observation"))
    require(isinstance(rec.get("offset_ns"), int), "offset_ns must be int")
    require(not isinstance(rec.get("offset_ns"), bool), "offset_ns must be int")
    end_offset = rec.get("end_offset_ns")
    if end_offset is not None:
        require(isinstance(end_offset, int), "end_offset_ns must be int or null")
        require(not isinstance(end_offset, bool), "end_offset_ns must be int or null")
    check_drift(rec.get("drift_ns_per_s"))
    check_clock_resolution(rec.get("clock_resolution"))
    require_nonneg_int(rec.get("assumed_utc_bound_ns"), "assumed_utc_bound_ns")
    discontinuities = rec.get("discontinuities")
    require(isinstance(discontinuities, list), "discontinuities must be a list")
    for item in discontinuities:
        check_discontinuity_record(item)
    check_validity(rec.get("validity"))
    check_join_order(rec.get("join_order"))
    check_cross_policy(rec.get("correlation_across_discontinuity"))


def check_clock_skew_record(rec: dict[str, Any]) -> None:
    require(rec.get("schema_version") == SCHEMA, f"unexpected schema_version: {rec.get('schema_version')}")
    require(rec.get("record_kind") == CLOCK_SKEW, f"expected record_kind {CLOCK_SKEW}")
    require_nonempty_str(rec.get("agoge_run_id"), "agoge_run_id required")
    host = rec.get("host")
    require(isinstance(host, dict), "host object required")
    require_nonempty_str(host.get("hostname"), "host.hostname required")
    require(isinstance(rec.get("gpu"), dict), "gpu identity object required")
    parse_rfc3339_utc(rec.get("timestamp_utc"))
    require_nonneg_int(rec.get("monotonic_ns"), "monotonic_ns must be a non-negative int")
    collector = rec.get("collector")
    require(isinstance(collector, dict), "collector object required")
    require_nonempty_str(collector.get("id"), "collector.id must be a non-empty string")
    require_nonempty_str(collector.get("version"), "collector.version must be a non-empty string")
    require(rec.get("cadence_ms") is None, "bkl_clock_skew cadence_ms must be null")
    check_clock_skew_payload(rec)


def build_clock_skew_record(
    report: dict[str, Any],
    *,
    agoge_run_id: str,
    host: dict[str, Any],
    gpu: dict[str, Any],
    collector: dict[str, str],
) -> dict[str, Any]:
    start = report["start_observation"]
    record = {
        "schema_version": SCHEMA,
        "record_kind": CLOCK_SKEW,
        "agoge_run_id": agoge_run_id,
        "host": host,
        "gpu": gpu,
        "timestamp_utc": start["timestamp_utc"],
        "monotonic_ns": start["monotonic_ns"],
        "collector": collector,
        "cadence_ms": None,
    }
    record.update(report)
    return record
