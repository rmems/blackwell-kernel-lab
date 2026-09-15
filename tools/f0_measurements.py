"""Shared F0 measurement objects: missingness is never a fabricated zero.

Used by the correlation join validator and NVML capability discovery.
Physical quantities are objects with ``value``, ``unit``, and ``status``.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

SCHEMA = "bkl.f0_correlation.v1"
CAPABILITY_SCHEMA = "bkl.f0_capability.v1"
CAPABILITY_KIND = "bkl_gpu_capability"
MARKER = "agoge_marker"
SAMPLE = "bkl_gpu_sample"

# ok is a real reading (zero allowed). Every other status requires value=null.
MEASUREMENT_STATUS = frozenset(
    {
        "ok",
        "unavailable",
        "unsupported",
        "permission_denied",
        "transient_failure",
        "device_lost",
    }
)
MISSING_STATUSES = MEASUREMENT_STATUS - {"ok"}
DEVICE_STATUS = frozenset(
    {
        "ok",
        "no_device",
        "device_lost",
        "permission_denied",
        "transient_failure",
        "unsupported",
    }
)
CAPABILITY_STATUS = frozenset(
    {
        "supported",
        "unsupported",
        "permission_denied",
        "device_lost",
    }
)
DIGEST_PREFIX = "sha256:"
DIGEST_HEX_LEN = 64

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
PERCENT_FIELDS = frozenset({"utilization_gpu", "utilization_memory"})
NUMERIC_METRIC_NAMES = tuple(name for name, _unit in PHYSICAL)
THROTTLE_METRIC = "throttle"
PSTATE_METRIC = "performance_state"
PSTATE_UNIT = "pstate"
CAPABILITY_METRICS = NUMERIC_METRIC_NAMES + (THROTTLE_METRIC, PSTATE_METRIC)

# Probe status → whether the sensor exists for later sampling (#53).
STATUS_TO_CAPABILITY = {
    "ok": "supported",
    "unavailable": "supported",
    "transient_failure": "supported",
    "unsupported": "unsupported",
    "permission_denied": "permission_denied",
    "device_lost": "device_lost",
}


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


def check_ok_numeric(value: Any, name: str) -> None:
    require(isinstance(value, (int, float)), f"{name}: ok requires a numeric value, not {value!r}")
    require(not isinstance(value, bool), f"{name}: ok requires a numeric value, not {value!r}")
    require(math.isfinite(float(value)), f"{name}: ok requires a finite number, not {value!r}")


def check_measurement(obj: Any, name: str, unit: str) -> None:
    require(isinstance(obj, dict), f"{name} must be a measurement object")
    require("value" in obj, f"{name}: value key required")
    require(obj.get("unit") == unit, f"{name}: expected unit {unit}, got {obj.get('unit')}")
    status = obj.get("status")
    require(status in MEASUREMENT_STATUS, f"{name}: bad status {status}")
    value = obj["value"]
    if status == "ok":
        check_ok_numeric(value, name)
        return
    require(value is None, f"{name}: missing must be null, not {value!r} (missing ≠ zero)")


def check_percent_bounds(obj: Mapping[str, Any], name: str) -> None:
    if obj.get("status") != "ok":
        return
    value = obj["value"]
    require(0 <= value <= 100, f"{name}: ok percent must be in 0–100, got {value!r}")


def check_throttle_reasons(status: str, reasons: Any) -> None:
    if status == "ok":
        require(isinstance(reasons, list), "throttle.reasons must be a list when ok")
        require(all(isinstance(item, str) for item in reasons), "throttle.reasons must be strings")
        return
    if reasons is None:
        return
    require(isinstance(reasons, list), "throttle.reasons must be null or a list when not ok")
    require(len(reasons) == 0, "throttle.reasons must be null or empty when not ok")


def check_throttle(obj: Any) -> None:
    require(isinstance(obj, dict), "throttle must be an object")
    status = obj.get("status")
    require(status in MEASUREMENT_STATUS, f"throttle: bad status {status}")
    check_throttle_reasons(status, obj.get("reasons"))


def require_capability_digest(value: Any, message: str) -> None:
    require_nonempty_str(value, message)
    require(value.startswith(DIGEST_PREFIX), message)
    digest_hex = value[len(DIGEST_PREFIX) :]
    require(len(digest_hex) == DIGEST_HEX_LEN, message)
    require(all(char in "0123456789abcdef" for char in digest_hex), message)


def capability_for_status(status: str) -> str:
    mapped = STATUS_TO_CAPABILITY.get(status)
    require(mapped in CAPABILITY_STATUS, f"unhandled measurement status {status!r}")
    return mapped


def measurement(status: str, unit: str, value: Any = None) -> dict[str, Any]:
    """Build a measurement object. Non-ok statuses force value to null."""
    require(status in MEASUREMENT_STATUS, f"bad status {status}")
    if status == "ok":
        check_ok_numeric(value, "measurement")
        return {"value": value, "unit": unit, "status": status}
    return {"value": None, "unit": unit, "status": status}
