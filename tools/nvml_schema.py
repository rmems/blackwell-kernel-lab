"""Schema checks for F0 NVML capability snapshots and run binding."""

from __future__ import annotations

from typing import Any

from f0_measurements import (
    CAPABILITY_KIND,
    CAPABILITY_SCHEMA,
    CAPABILITY_STATUS,
    DEVICE_STATUS,
    PERCENT_FIELDS,
    PHYSICAL,
    PSTATE_METRIC,
    PSTATE_UNIT,
    SchemaError,
    THROTTLE_METRIC,
    capability_for_status,
    check_measurement,
    check_percent_bounds,
    check_throttle,
    parse_rfc3339_utc,
    require,
    require_capability_digest,
    require_nonempty_str,
    require_nonneg_int,
    require_optional_nonempty_str,
)
from nvml_capability import capability_digest


GPU_IDENTITY_KEYS = ("pci_bus_id", "uuid", "name", "compute_capability")
TOOL_KEYS = ("nvml_version", "driver_version", "nvidia_smi")


def check_nullable_str_fields(obj: Any, keys: tuple[str, ...], prefix: str) -> None:
    require(isinstance(obj, dict), f"{prefix} object required")
    for key in keys:
        require(key in obj, f"{prefix}.{key} required (null if unknown)")
        require_optional_nonempty_str(obj.get(key), f"{prefix}.{key} must be null or a non-empty string")


def check_gpu_identity_fields(gpu: Any) -> None:
    check_nullable_str_fields(gpu, GPU_IDENTITY_KEYS, "gpu")


def check_tools(tools: Any) -> None:
    check_nullable_str_fields(tools, TOOL_KEYS, "tools")


def check_numeric_capability(rec: Any, name: str, unit: str) -> None:
    require(isinstance(rec, dict), f"{name} must be an object")
    require(rec.get("capability") in CAPABILITY_STATUS, f"{name}: bad capability {rec.get('capability')}")
    check_measurement(rec, name, unit)
    if name in PERCENT_FIELDS:
        check_percent_bounds(rec, name)
    expected = capability_for_status(rec["status"])
    require(rec["capability"] == expected, f"{name}: capability {rec['capability']} != {expected}")


def check_throttle_capability(rec: Any) -> None:
    require(isinstance(rec, dict), "throttle must be an object")
    check_throttle(rec)
    require(rec.get("capability") in CAPABILITY_STATUS, f"throttle: bad capability {rec.get('capability')}")
    expected = capability_for_status(rec["status"])
    require(rec["capability"] == expected, f"throttle: capability {rec['capability']} != {expected}")
    require(rec.get("value") is None, "throttle value must be null")
    require(rec.get("unit") is None, "throttle unit must be null")


def check_metrics_block(metrics: Any) -> None:
    require(isinstance(metrics, dict), "metrics object required")
    for name, unit in PHYSICAL:
        require(name in metrics, f"missing metric {name}")
        check_numeric_capability(metrics[name], name, unit)
    require(PSTATE_METRIC in metrics, "missing metric performance_state")
    check_numeric_capability(metrics[PSTATE_METRIC], PSTATE_METRIC, PSTATE_UNIT)
    require(THROTTLE_METRIC in metrics, "missing metric throttle")
    check_throttle_capability(metrics[THROTTLE_METRIC])


def check_device_identity_rule(rec: dict[str, Any]) -> None:
    if rec["device_status"] != "ok":
        return
    gpu = rec["gpu"]
    has_id = bool(gpu.get("uuid") or gpu.get("pci_bus_id"))
    require(has_id, "device_status=ok requires gpu.uuid or gpu.pci_bus_id")


def check_capability_time(rec: dict[str, Any]) -> None:
    parse_rfc3339_utc(rec.get("timestamp_utc"))
    require_nonneg_int(rec.get("monotonic_ns"), "monotonic_ns must be a non-negative int")


def check_capability_collector(rec: dict[str, Any]) -> None:
    collector = rec.get("collector")
    require(isinstance(collector, dict), "collector object required")
    require_nonempty_str(collector.get("id"), "collector.id required")
    require_nonempty_str(collector.get("version"), "collector.version required")


def check_capability_envelope(rec: dict[str, Any]) -> None:
    require(rec.get("schema_version") == CAPABILITY_SCHEMA, f"unexpected schema_version: {rec.get('schema_version')}")
    require(rec.get("record_kind") == CAPABILITY_KIND, f"expected {CAPABILITY_KIND}")
    require_nonempty_str(rec.get("agoge_run_id"), "agoge_run_id required")
    host = rec.get("host")
    require(isinstance(host, dict), "host object required")
    require_nonempty_str(host.get("hostname"), "host.hostname required")
    check_gpu_identity_fields(rec.get("gpu"))
    check_capability_time(rec)
    check_capability_collector(rec)


def check_capability_snapshot(rec: Any) -> None:
    require(isinstance(rec, dict), "capability snapshot must be an object")
    check_capability_envelope(rec)
    check_tools(rec.get("tools"))
    device_status = rec.get("device_status")
    require(device_status in DEVICE_STATUS, f"bad device_status {device_status}")
    check_metrics_block(rec.get("metrics"))
    require_capability_digest(rec.get("capability_digest"), "capability_digest must be sha256:<64 hex>")
    require(rec["capability_digest"] == capability_digest(rec), "capability_digest does not match snapshot")
    check_device_identity_rule(rec)


def nonempty_id(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def gpu_identity_matches(cap_gpu: Any, sample_gpu: Any) -> bool:
    if not isinstance(cap_gpu, dict) or not isinstance(sample_gpu, dict):
        return False
    cap_uuid = nonempty_id(cap_gpu.get("uuid"))
    sample_uuid = nonempty_id(sample_gpu.get("uuid"))
    if cap_uuid is not None and sample_uuid is not None:
        return cap_uuid == sample_uuid
    cap_pci = nonempty_id(cap_gpu.get("pci_bus_id"))
    sample_pci = nonempty_id(sample_gpu.get("pci_bus_id"))
    if cap_pci is not None and sample_pci is not None:
        return cap_pci == sample_pci
    return False


def check_bind_host_gpu(snapshot: dict[str, Any], sample: dict[str, Any], index: int) -> None:
    host = sample.get("host")
    require(isinstance(host, dict), f"sample {index} host object required")
    require(host.get("hostname") == snapshot["host"]["hostname"], f"sample {index} host mismatch")
    if snapshot["device_status"] != "ok":
        return
    require(
        gpu_identity_matches(snapshot["gpu"], sample.get("gpu")),
        f"sample {index} GPU identity mismatch",
    )


def check_discovery_precedes_sample(snapshot: dict[str, Any], sample: dict[str, Any], index: int) -> None:
    require(
        snapshot["monotonic_ns"] <= sample["monotonic_ns"],
        f"sample {index} monotonic_ns precedes capability discovery",
    )
    cap_time = parse_rfc3339_utc(snapshot["timestamp_utc"])
    sample_time = parse_rfc3339_utc(sample["timestamp_utc"])
    require(cap_time <= sample_time, f"sample {index} is timestamped before capability discovery")


def bind_one_sample(snapshot: dict[str, Any], sample: Any, index: int, digest: str, run_id: str) -> None:
    require(isinstance(sample, dict), f"sample {index} must be an object")
    require(sample.get("agoge_run_id") == run_id, f"sample {index} agoge_run_id mismatch")
    require_capability_digest(
        sample.get("capability_digest"),
        f"sample {index} capability_digest must match the run snapshot",
    )
    require(sample["capability_digest"] == digest, f"sample {index} capability_digest mismatch")
    check_bind_host_gpu(snapshot, sample, index)
    check_discovery_precedes_sample(snapshot, sample, index)


def bind_samples(snapshot: dict[str, Any], samples: list[dict[str, Any]]) -> None:
    check_capability_snapshot(snapshot)
    digest = snapshot["capability_digest"]
    run_id = snapshot["agoge_run_id"]
    require(samples, "no samples to bind")
    for index, sample in enumerate(samples):
        bind_one_sample(snapshot, sample, index, digest, run_id)


def reject_fabricated_zero(status: str) -> None:
    """Regression: unavailable/denied/lost must not encode as numeric zero."""
    bad = {"value": 0, "unit": "W", "status": status}
    try:
        check_measurement(bad, "power", "W")
    except SchemaError:
        return
    raise SchemaError(f"{status} power encoded as 0 must be rejected")
