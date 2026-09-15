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
    require,
    require_capability_digest,
    require_nonempty_str,
    require_nonneg_int,
    require_optional_nonempty_str,
)
from nvml_capability import capability_digest


def check_gpu_identity_fields(gpu: Any) -> None:
    require(isinstance(gpu, dict), "gpu identity object required")
    require_optional_nonempty_str(gpu.get("pci_bus_id"), "gpu.pci_bus_id must be null or a non-empty string")
    require_optional_nonempty_str(gpu.get("uuid"), "gpu.uuid must be null or a non-empty string")
    require_optional_nonempty_str(gpu.get("name"), "gpu.name must be null or a non-empty string")
    require_optional_nonempty_str(
        gpu.get("compute_capability"),
        "gpu.compute_capability must be null or a non-empty string",
    )


def check_tools(tools: Any) -> None:
    require(isinstance(tools, dict), "tools object required")
    require_optional_nonempty_str(tools.get("nvml_version"), "tools.nvml_version must be null or a string")
    require_optional_nonempty_str(tools.get("driver_version"), "tools.driver_version must be null or a string")
    require_optional_nonempty_str(tools.get("nvidia_smi"), "tools.nvidia_smi must be null or a string")


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
    require_nonempty_str(rec.get("timestamp_utc"), "timestamp_utc required")
    require(str(rec.get("timestamp_utc", "")).endswith("Z"), "timestamp_utc must end in Z")
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


def bind_samples(snapshot: dict[str, Any], samples: list[dict[str, Any]]) -> None:
    check_capability_snapshot(snapshot)
    digest = snapshot["capability_digest"]
    run_id = snapshot["agoge_run_id"]
    require(samples, "no samples to bind")
    for index, sample in enumerate(samples):
        require(isinstance(sample, dict), f"sample {index} must be an object")
        require(sample.get("agoge_run_id") == run_id, f"sample {index} agoge_run_id mismatch")
        require_capability_digest(
            sample.get("capability_digest"),
            f"sample {index} capability_digest must match the run snapshot",
        )
        require(sample["capability_digest"] == digest, f"sample {index} capability_digest mismatch")


def reject_fabricated_zero(status: str) -> None:
    """Regression: unavailable/denied/lost must not encode as numeric zero."""
    bad = {"value": 0, "unit": "W", "status": status}
    try:
        check_measurement(bad, "power", "W")
    except SchemaError:
        return
    raise SchemaError(f"{status} power encoded as 0 must be rejected")
