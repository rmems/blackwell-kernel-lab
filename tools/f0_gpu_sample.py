"""Build one ``bkl_gpu_sample`` from an NVML backend reading.

Missing probe results stay ``unavailable`` / ``unsupported`` /
``permission_denied`` / ``transient_failure`` / ``device_lost`` with a null
value. This module does not loop, sleep, or start a CUDA context.
"""

from __future__ import annotations

from typing import Any

from f0_measurements import (
    PERCENT_FIELDS,
    PHYSICAL,
    PSTATE_METRIC,
    PSTATE_UNIT,
    SAMPLE,
    SCHEMA,
    SchemaError,
    measurement,
)
from nvml_capability import capture_probe

COLLECTOR_ID = "bkl-gpu-sampler"
COLLECTOR_VERSION = "0.1.0"
MS_TO_NS = 1_000_000


def copy_gpu(gpu: dict[str, Any]) -> dict[str, Any]:
    return {
        "pci_bus_id": gpu.get("pci_bus_id"),
        "uuid": gpu.get("uuid"),
        "name": gpu.get("name"),
        "compute_capability": gpu.get("compute_capability"),
    }


def cuda_from_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    tools = snapshot.get("tools")
    driver = None
    if isinstance(tools, dict):
        driver = tools.get("driver_version")
    return {"driver_version": driver, "runtime_version": None, "tool": "nvml"}


def interval_ns(monotonic_ns: int, previous_ns: int | None, cadence_ms: int) -> tuple[int | None, int | None]:
    if previous_ns is None:
        return None, None
    observed = monotonic_ns - previous_ns
    return observed, observed - cadence_ms * MS_TO_NS


def percent_in_range(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return 0 <= float(value) <= 100


def bounded_percent(name: str, record: dict[str, Any]) -> dict[str, Any]:
    if name not in PERCENT_FIELDS or record.get("status") != "ok":
        return record
    if percent_in_range(record.get("value")):
        return record
    return measurement("transient_failure", "%")


def safe_ok(unit: str, value: Any) -> dict[str, Any]:
    try:
        return measurement("ok", unit, value)
    except SchemaError:
        return measurement("transient_failure", unit)


def read_named(backend: Any, index: int, name: str, unit: str) -> dict[str, Any]:
    status, value, _detail = capture_probe(lambda: backend.read_numeric(index, name), "transient_failure")
    if status != "ok":
        return measurement(status, unit)
    return bounded_percent(name, safe_ok(unit, value))


def read_physical(backend: Any, index: int) -> dict[str, dict[str, Any]]:
    return {name: read_named(backend, index, name, unit) for name, unit in PHYSICAL}


def throttle_ok(value: Any) -> dict[str, Any] | None:
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return {"status": "ok", "reasons": list(value)}
    return None


def read_throttle_sample(backend: Any, index: int) -> dict[str, Any]:
    status, value, _detail = capture_probe(lambda: backend.read_throttle(index), "transient_failure")
    if status != "ok":
        return {"status": status, "reasons": None}
    accepted = throttle_ok(value)
    if accepted is not None:
        return accepted
    return {"status": "transient_failure", "reasons": None}


def read_performance_state(backend: Any, index: int) -> dict[str, Any]:
    return read_named(backend, index, PSTATE_METRIC, PSTATE_UNIT)


def build_gpu_sample(
    snapshot: dict[str, Any],
    *,
    cadence_ms: int,
    timestamp_utc: str,
    monotonic_ns: int,
    physical: dict[str, dict[str, Any]],
    throttle: dict[str, Any],
    performance_state: dict[str, Any],
    collect_duration_ns: int,
    previous_ns: int | None,
) -> dict[str, Any]:
    observed, jitter = interval_ns(monotonic_ns, previous_ns, cadence_ms)
    sample: dict[str, Any] = {
        "schema_version": SCHEMA,
        "record_kind": SAMPLE,
        "agoge_run_id": snapshot["agoge_run_id"],
        "host": {"hostname": snapshot["host"]["hostname"]},
        "gpu": copy_gpu(snapshot["gpu"]),
        "timestamp_utc": timestamp_utc,
        "monotonic_ns": monotonic_ns,
        "collector": {"id": COLLECTOR_ID, "version": COLLECTOR_VERSION},
        "cadence_ms": cadence_ms,
        "throttle": throttle,
        "profile_window_ref": None,
        "cuda": cuda_from_snapshot(snapshot),
        "capability_digest": snapshot["capability_digest"],
        "performance_state": performance_state,
        "collect_duration_ns": collect_duration_ns,
        "observed_interval_ns": observed,
        "jitter_ns": jitter,
    }
    sample.update(physical)
    return sample

