"""NVML capability discovery for F0 GPU telemetry (CPU-testable).

Discovers which F0 sampler metrics a backend can expose and encodes
missingness without substituting zeroes. Fake backends live in nvml_fakes.py;
schema checks live in nvml_schema.py. Live NVML is optional.

This is not a sampler loop (#53), a daemon, or a safety controller.
"""

from __future__ import annotations

import hashlib
import json
import socket
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Protocol

from f0_measurements import (
    CAPABILITY_KIND,
    CAPABILITY_METRICS,
    CAPABILITY_SCHEMA,
    DEVICE_STATUS,
    DIGEST_PREFIX,
    MEASUREMENT_STATUS,
    NUMERIC_METRIC_NAMES,
    PHYSICAL,
    PSTATE_METRIC,
    PSTATE_UNIT,
    SchemaError,
    THROTTLE_METRIC,
    capability_for_status,
    measurement,
    require,
)

COLLECTOR_ID = "bkl-nvml-capability"
COLLECTOR_VERSION = "0.1.0"
UNITS = dict(PHYSICAL)
UNITS[PSTATE_METRIC] = PSTATE_UNIT

NVML_SUCCESS = 0
NVML_ERROR_INVALID_ARGUMENT = 2
NVML_ERROR_NOT_SUPPORTED = 3
NVML_ERROR_NO_PERMISSION = 4
NVML_ERROR_NOT_FOUND = 6
NVML_ERROR_DRIVER_NOT_LOADED = 9
NVML_ERROR_TIMEOUT = 10
NVML_ERROR_LIBRARY_NOT_FOUND = 12
NVML_ERROR_FUNCTION_NOT_FOUND = 13
NVML_ERROR_GPU_IS_LOST = 15
NVML_ERROR_RESET_REQUIRED = 16
NVML_ERROR_NO_DATA = 21
NVML_ERROR_NOT_READY = 27
NVML_ERROR_GPU_NOT_FOUND = 28

NVML_METRIC_STATUS = {
    NVML_ERROR_INVALID_ARGUMENT: "unsupported",
    NVML_ERROR_NOT_SUPPORTED: "unsupported",
    NVML_ERROR_NO_PERMISSION: "permission_denied",
    NVML_ERROR_NOT_FOUND: "unsupported",
    NVML_ERROR_DRIVER_NOT_LOADED: "unsupported",
    NVML_ERROR_TIMEOUT: "transient_failure",
    NVML_ERROR_LIBRARY_NOT_FOUND: "unsupported",
    NVML_ERROR_FUNCTION_NOT_FOUND: "unsupported",
    NVML_ERROR_GPU_IS_LOST: "device_lost",
    NVML_ERROR_RESET_REQUIRED: "transient_failure",
    NVML_ERROR_NO_DATA: "unavailable",
    NVML_ERROR_NOT_READY: "transient_failure",
    NVML_ERROR_GPU_NOT_FOUND: "device_lost",
}

THROTTLE_REASON_BITS = (
    (0x1, "gpu_idle"),
    (0x2, "applications_clocks_setting"),
    (0x4, "sw_power_cap"),
    (0x8, "hw_slowdown"),
    (0x10, "sync_boost"),
    (0x20, "sw_thermal_slowdown"),
    (0x40, "hw_thermal_slowdown"),
    (0x80, "hw_power_brake_slowdown"),
    (0x100, "display_clock_setting"),
)


class ProbeFailure(Exception):
    """A single metric or identity probe failed. Discovery continues."""

    def __init__(self, status: str, detail: str | None = None) -> None:
        if status not in MEASUREMENT_STATUS and status not in DEVICE_STATUS:
            raise SchemaError(f"bad probe status {status!r}")
        self.status = status
        self.detail = detail
        super().__init__(detail or status)


@dataclass(frozen=True)
class GpuIdentity:
    pci_bus_id: str | None
    uuid: str | None
    name: str | None
    compute_capability: str | None


@dataclass(frozen=True)
class ToolVersions:
    nvml_version: str | None
    driver_version: str | None
    nvidia_smi: str | None


@dataclass(frozen=True)
class ProbeContext:
    agoge_run_id: str
    hostname: str
    timestamp_utc: str
    monotonic_ns: int
    collector_id: str = COLLECTOR_ID
    collector_version: str = COLLECTOR_VERSION


class NvmlBackend(Protocol):
    def gpu_count(self) -> int:
        """Return attached GPU count. Raises ProbeFailure on init/device loss."""

    def identity(self, index: int) -> GpuIdentity:
        """Stable UUID/PCI identity for GPU ``index``."""

    def versions(self) -> ToolVersions:
        """NVML/driver/tool versions. Missing tools are null, not empty."""

    def read_numeric(self, index: int, metric: str) -> int | float:
        """Return a finite numeric reading or raise ProbeFailure."""

    def read_throttle(self, index: int) -> list[str]:
        """Return throttle reason names or raise ProbeFailure."""


def map_nvml_error(code: int) -> str:
    if code == NVML_SUCCESS:
        return "ok"
    return NVML_METRIC_STATUS.get(code, "transient_failure")


def decode_throttle_reasons(mask: int) -> list[str]:
    return [name for bit, name in THROTTLE_REASON_BITS if mask & bit]


def utc_now_z() -> str:
    stamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    return stamp.replace("+00:00", "Z")


def monotonic_ns_now() -> int:
    return time.monotonic_ns()


def empty_identity() -> GpuIdentity:
    return GpuIdentity(pci_bus_id=None, uuid=None, name=None, compute_capability=None)


def identity_dict(identity: GpuIdentity) -> dict[str, str | None]:
    return {
        "pci_bus_id": identity.pci_bus_id,
        "uuid": identity.uuid,
        "name": identity.name,
        "compute_capability": identity.compute_capability,
    }


def versions_dict(versions: ToolVersions) -> dict[str, str | None]:
    return {
        "nvml_version": versions.nvml_version,
        "driver_version": versions.driver_version,
        "nvidia_smi": versions.nvidia_smi,
    }


def numeric_metric_record(status: str, unit: str, value: Any = None) -> dict[str, Any]:
    if status == "ok":
        body = measurement(status, unit, value)
    else:
        body = measurement(status, unit, None)
    body["capability"] = capability_for_status(status)
    return body


def throttle_metric_record(status: str, reasons: list[str] | None = None) -> dict[str, Any]:
    if status == "ok":
        return {
            "capability": "supported",
            "status": "ok",
            "reasons": list(reasons or []),
            "value": None,
            "unit": None,
        }
    return {
        "capability": capability_for_status(status),
        "status": status,
        "reasons": None,
        "value": None,
        "unit": None,
    }


def capture_probe(call: Callable[[], Any], fallback: str) -> tuple[str, Any, str | None]:
    """Run one probe. Never raises; optional metric failure is a status."""
    try:
        return "ok", call(), None
    except ProbeFailure as error:
        status = error.status if error.status in MEASUREMENT_STATUS else fallback
        if status == "ok":
            status = fallback
        return status, None, error.detail
    except (OSError, RuntimeError, ValueError, TypeError, AttributeError) as error:
        return "transient_failure", None, str(error)


def digest_payload(snapshot: dict[str, Any]) -> dict[str, Any]:
    metrics = snapshot["metrics"]
    return {
        "schema_version": snapshot["schema_version"],
        "device_status": snapshot["device_status"],
        "gpu": snapshot["gpu"],
        "tools": snapshot["tools"],
        "metrics": {
            name: {
                "capability": metrics[name]["capability"],
                "unit": metrics[name].get("unit"),
            }
            for name in CAPABILITY_METRICS
        },
    }


def capability_digest(snapshot: dict[str, Any]) -> str:
    blob = json.dumps(digest_payload(snapshot), sort_keys=True, separators=(",", ":")).encode()
    return DIGEST_PREFIX + hashlib.sha256(blob).hexdigest()


def attach_digest(snapshot: dict[str, Any]) -> dict[str, Any]:
    snapshot["capability_digest"] = capability_digest(snapshot)
    return snapshot


def missing_metrics(status: str) -> dict[str, Any]:
    records = {name: numeric_metric_record(status, UNITS[name]) for name in NUMERIC_METRIC_NAMES}
    records[THROTTLE_METRIC] = throttle_metric_record(status)
    records[PSTATE_METRIC] = numeric_metric_record(status, PSTATE_UNIT)
    return records


def snapshot_with(
    ctx: ProbeContext,
    identity: GpuIdentity,
    versions: ToolVersions,
    device_status: str,
    metrics: dict[str, Any],
    notes: dict[str, str] | None = None,
) -> dict[str, Any]:
    rec = {
        "schema_version": CAPABILITY_SCHEMA,
        "record_kind": CAPABILITY_KIND,
        "agoge_run_id": ctx.agoge_run_id,
        "host": {"hostname": ctx.hostname},
        "gpu": identity_dict(identity),
        "timestamp_utc": ctx.timestamp_utc,
        "monotonic_ns": ctx.monotonic_ns,
        "collector": {"id": ctx.collector_id, "version": ctx.collector_version},
        "tools": versions_dict(versions),
        "device_status": device_status,
        "metrics": metrics,
    }
    if notes:
        rec["metric_notes"] = notes
    return attach_digest(rec)


def try_count(backend: NvmlBackend) -> tuple[str, int | None, str | None]:
    status, value, detail = capture_probe(backend.gpu_count, "transient_failure")
    if status != "ok":
        mapped = status if status in DEVICE_STATUS else "transient_failure"
        return mapped, None, detail
    require(isinstance(value, int) and not isinstance(value, bool) and value >= 0, "gpu_count must be >= 0")
    return status, value, detail


def try_identity(backend: NvmlBackend, index: int) -> tuple[GpuIdentity, str | None, str | None]:
    status, value, detail = capture_probe(lambda: backend.identity(index), "device_lost")
    if status == "ok" and isinstance(value, GpuIdentity):
        return value, None, detail
    return empty_identity(), status, detail


def try_versions(backend: NvmlBackend) -> ToolVersions:
    status, value, _detail = capture_probe(backend.versions, "unsupported")
    if status == "ok" and isinstance(value, ToolVersions):
        return value
    return ToolVersions(nvml_version=None, driver_version=None, nvidia_smi=None)


def try_numeric(backend: NvmlBackend, index: int, name: str) -> tuple[dict[str, Any], str | None]:
    unit = UNITS[name]
    status, value, detail = capture_probe(lambda: backend.read_numeric(index, name), "transient_failure")
    if status == "ok":
        return numeric_metric_record("ok", unit, value), detail
    return numeric_metric_record(status, unit), detail


def try_throttle(backend: NvmlBackend, index: int) -> tuple[dict[str, Any], str | None]:
    status, value, detail = capture_probe(lambda: backend.read_throttle(index), "transient_failure")
    if status != "ok":
        return throttle_metric_record(status), detail
    require(isinstance(value, list), "throttle reasons must be a list")
    return throttle_metric_record("ok", value), detail


def note_if(notes: dict[str, str], key: str, detail: str | None) -> None:
    if detail:
        notes[key] = detail


def discover_metrics(backend: NvmlBackend, index: int) -> tuple[dict[str, Any], dict[str, str]]:
    """Probe every F0 field. One failure never aborts the remaining probes."""
    metrics: dict[str, Any] = {}
    notes: dict[str, str] = {}
    for name in NUMERIC_METRIC_NAMES:
        record, detail = try_numeric(backend, index, name)
        metrics[name] = record
        note_if(notes, name, detail)
    throttle, throttle_detail = try_throttle(backend, index)
    metrics[THROTTLE_METRIC] = throttle
    note_if(notes, THROTTLE_METRIC, throttle_detail)
    pstate, pstate_detail = try_numeric(backend, index, PSTATE_METRIC)
    metrics[PSTATE_METRIC] = pstate
    note_if(notes, PSTATE_METRIC, pstate_detail)
    return metrics, notes


def failed_count_snapshot(
    ctx: ProbeContext,
    versions: ToolVersions,
    count_status: str,
    notes: dict[str, str],
) -> dict[str, Any]:
    metric_status = "unsupported" if count_status == "unsupported" else count_status
    if metric_status not in MEASUREMENT_STATUS:
        metric_status = "transient_failure"
    device_status = count_status if count_status in DEVICE_STATUS else "transient_failure"
    return snapshot_with(ctx, empty_identity(), versions, device_status, missing_metrics(metric_status), notes or None)


def resolve_device_status(identity_status: str | None, metrics: dict[str, Any]) -> str:
    if identity_status is None:
        device_status = "ok"
    elif identity_status in DEVICE_STATUS:
        device_status = identity_status
    else:
        device_status = "transient_failure"
    if device_status == "ok" and any(rec["status"] == "device_lost" for rec in metrics.values()):
        return "device_lost"
    return device_status


def make_probe_context(
    agoge_run_id: str,
    hostname: str | None,
    timestamp_utc: str | None,
    monotonic_ns: int | None,
    collector_id: str,
    collector_version: str,
) -> ProbeContext:
    host = hostname if hostname is not None else socket.gethostname()
    stamp = timestamp_utc if timestamp_utc is not None else utc_now_z()
    mono = monotonic_ns if monotonic_ns is not None else monotonic_ns_now()
    return ProbeContext(
        agoge_run_id=agoge_run_id,
        hostname=host,
        timestamp_utc=stamp,
        monotonic_ns=mono,
        collector_id=collector_id,
        collector_version=collector_version,
    )


def discover_present_gpu(
    backend: NvmlBackend,
    ctx: ProbeContext,
    versions: ToolVersions,
    notes: dict[str, str],
    gpu_index: int,
) -> dict[str, Any]:
    identity, identity_status, identity_detail = try_identity(backend, gpu_index)
    note_if(notes, "identity", identity_detail)
    metrics, metric_notes = discover_metrics(backend, gpu_index)
    notes.update(metric_notes)
    status = resolve_device_status(identity_status, metrics)
    return snapshot_with(ctx, identity, versions, status, metrics, notes or None)


def discover(
    backend: NvmlBackend,
    *,
    agoge_run_id: str,
    hostname: str | None = None,
    timestamp_utc: str | None = None,
    monotonic_ns: int | None = None,
    collector_id: str = COLLECTOR_ID,
    collector_version: str = COLLECTOR_VERSION,
    gpu_index: int = 0,
) -> dict[str, Any]:
    """Return a capability snapshot. Does not raise on optional metric failure."""
    ctx = make_probe_context(agoge_run_id, hostname, timestamp_utc, monotonic_ns, collector_id, collector_version)
    versions = try_versions(backend)
    count_status, count, count_detail = try_count(backend)
    notes: dict[str, str] = {}
    note_if(notes, "gpu_count", count_detail)
    if count_status != "ok":
        return failed_count_snapshot(ctx, versions, count_status, notes)
    if count == 0:
        return snapshot_with(ctx, empty_identity(), versions, "no_device", missing_metrics("unsupported"), notes or None)
    return discover_present_gpu(backend, ctx, versions, notes, gpu_index)
