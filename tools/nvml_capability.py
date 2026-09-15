"""NVML capability discovery for F0 GPU telemetry (CPU-testable).

Discovers which F0 sampler metrics a backend can expose, encodes missingness
without substituting zeroes, and binds a capability digest to a run. Live NVML
is optional; CPU CI uses fake backends.

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
    CAPABILITY_STATUS,
    DEVICE_STATUS,
    DIGEST_PREFIX,
    MEASUREMENT_STATUS,
    NUMERIC_METRIC_NAMES,
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
    measurement,
    require,
    require_capability_digest,
    require_nonempty_str,
    require_nonneg_int,
    require_optional_nonempty_str,
)

COLLECTOR_ID = "bkl-nvml-capability"
COLLECTOR_VERSION = "0.1.0"
UNITS = dict(PHYSICAL)
UNITS[PSTATE_METRIC] = PSTATE_UNIT

NVML_SUCCESS = 0
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
    NVML_ERROR_NOT_SUPPORTED: "unsupported",
    NVML_ERROR_NO_PERMISSION: "permission_denied",
    NVML_ERROR_NOT_FOUND: "unsupported",
    NVML_ERROR_DRIVER_NOT_LOADED: "device_lost",
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

SCENARIO_ENVELOPE = {
    "timestamp_utc": "2026-09-15T04:00:00Z",
    "monotonic_ns": 0,
    "hostname": "ShipOfTheseus",
}


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
    *,
    agoge_run_id: str,
    hostname: str,
    identity: GpuIdentity,
    versions: ToolVersions,
    device_status: str,
    metrics: dict[str, Any],
    timestamp_utc: str,
    monotonic_ns: int,
    collector_id: str = COLLECTOR_ID,
    collector_version: str = COLLECTOR_VERSION,
    notes: dict[str, str] | None = None,
) -> dict[str, Any]:
    rec = {
        "schema_version": CAPABILITY_SCHEMA,
        "record_kind": CAPABILITY_KIND,
        "agoge_run_id": agoge_run_id,
        "host": {"hostname": hostname},
        "gpu": identity_dict(identity),
        "timestamp_utc": timestamp_utc,
        "monotonic_ns": monotonic_ns,
        "collector": {"id": collector_id, "version": collector_version},
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
    *,
    agoge_run_id: str,
    host: str,
    versions: ToolVersions,
    count_status: str,
    stamp: str,
    mono: int,
    collector_id: str,
    collector_version: str,
    notes: dict[str, str],
) -> dict[str, Any]:
    metric_status = "unsupported" if count_status == "unsupported" else count_status
    if metric_status not in MEASUREMENT_STATUS:
        metric_status = "transient_failure"
    device_status = count_status if count_status in DEVICE_STATUS else "transient_failure"
    return snapshot_with(
        agoge_run_id=agoge_run_id,
        hostname=host,
        identity=empty_identity(),
        versions=versions,
        device_status=device_status,
        metrics=missing_metrics(metric_status),
        timestamp_utc=stamp,
        monotonic_ns=mono,
        collector_id=collector_id,
        collector_version=collector_version,
        notes=notes or None,
    )


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
    host = hostname if hostname is not None else socket.gethostname()
    stamp = timestamp_utc if timestamp_utc is not None else utc_now_z()
    mono = monotonic_ns if monotonic_ns is not None else monotonic_ns_now()
    versions = try_versions(backend)
    count_status, count, count_detail = try_count(backend)
    notes: dict[str, str] = {}
    note_if(notes, "gpu_count", count_detail)
    if count_status != "ok":
        return failed_count_snapshot(
            agoge_run_id=agoge_run_id,
            host=host,
            versions=versions,
            count_status=count_status,
            stamp=stamp,
            mono=mono,
            collector_id=collector_id,
            collector_version=collector_version,
            notes=notes,
        )
    if count == 0:
        return snapshot_with(
            agoge_run_id=agoge_run_id,
            hostname=host,
            identity=empty_identity(),
            versions=versions,
            device_status="no_device",
            metrics=missing_metrics("unsupported"),
            timestamp_utc=stamp,
            monotonic_ns=mono,
            collector_id=collector_id,
            collector_version=collector_version,
            notes=notes or None,
        )
    identity, identity_status, identity_detail = try_identity(backend, gpu_index)
    note_if(notes, "identity", identity_detail)
    metrics, metric_notes = discover_metrics(backend, gpu_index)
    notes.update(metric_notes)
    return snapshot_with(
        agoge_run_id=agoge_run_id,
        hostname=host,
        identity=identity,
        versions=versions,
        device_status=resolve_device_status(identity_status, metrics),
        metrics=metrics,
        timestamp_utc=stamp,
        monotonic_ns=mono,
        collector_id=collector_id,
        collector_version=collector_version,
        notes=notes or None,
    )


@dataclass(frozen=True)
class FakeMetric:
    status: str
    value: int | float | None = None
    reasons: tuple[str, ...] | None = None
    detail: str | None = None


@dataclass(frozen=True)
class FakeScenario:
    name: str
    count: int
    identity: GpuIdentity
    versions: ToolVersions
    metrics: dict[str, FakeMetric]
    count_error: ProbeFailure | None = None


def rtx5080_identity() -> GpuIdentity:
    return GpuIdentity(
        pci_bus_id="0000:01:00.0",
        uuid="GPU-fixture-0001",
        name="NVIDIA GeForce RTX 5080",
        compute_capability="12.0",
    )


def fixture_tool_versions() -> ToolVersions:
    return ToolVersions(nvml_version="13.3.58", driver_version="610.43.03", nvidia_smi="610.43.03")


def ok_metrics(*, util_gpu: float, power: float, temp_mem: FakeMetric | None = None) -> dict[str, FakeMetric]:
    return {
        "power": FakeMetric("ok", power),
        "temperature_gpu": FakeMetric("ok", 32.0),
        "temperature_memory": temp_mem if temp_mem is not None else FakeMetric("ok", 40.0),
        "utilization_gpu": FakeMetric("ok", util_gpu),
        "utilization_memory": FakeMetric("ok", 12.0),
        "clock_graphics": FakeMetric("ok", 300),
        "clock_memory": FakeMetric("ok", 5000),
        "vram_used": FakeMetric("ok", 2100),
        "vram_free": FakeMetric("ok", 14203),
        "vram_total": FakeMetric("ok", 16303),
        "headroom": FakeMetric("ok", 14203),
        THROTTLE_METRIC: FakeMetric("ok", reasons=()),
        PSTATE_METRIC: FakeMetric("ok", 8),
    }


def status_all(status: str, detail: str) -> dict[str, FakeMetric]:
    records = {name: FakeMetric(status, detail=detail) for name in NUMERIC_METRIC_NAMES}
    records[THROTTLE_METRIC] = FakeMetric(status, detail=detail)
    records[PSTATE_METRIC] = FakeMetric(status, detail=detail)
    return records


def scenario_table() -> dict[str, FakeScenario]:
    identity = rtx5080_identity()
    versions = fixture_tool_versions()
    partial = ok_metrics(
        util_gpu=10.0,
        power=45.0,
        temp_mem=FakeMetric("unsupported", detail="NVML_ERROR_NOT_SUPPORTED"),
    )
    permission = ok_metrics(util_gpu=5.0, power=45.0)
    permission["power"] = FakeMetric("permission_denied", detail="NVML_ERROR_NO_PERMISSION")
    permission["temperature_gpu"] = FakeMetric("permission_denied", detail="NVML_ERROR_NO_PERMISSION")
    permission["temperature_memory"] = FakeMetric("permission_denied", detail="NVML_ERROR_NO_PERMISSION")
    transient = ok_metrics(util_gpu=20.0, power=90.0)
    transient["power"] = FakeMetric("transient_failure", detail="NVML_ERROR_TIMEOUT")
    transient["clock_graphics"] = FakeMetric("unavailable", detail="NVML_ERROR_NO_DATA")
    return {
        "full-support": FakeScenario(
            "full-support", 1, identity, versions, ok_metrics(util_gpu=47.0, power=180.0)
        ),
        "partial-support": FakeScenario("partial-support", 1, identity, versions, partial),
        "permission-denied": FakeScenario("permission-denied", 1, identity, versions, permission),
        "transient-failure": FakeScenario("transient-failure", 1, identity, versions, transient),
        "device-lost": FakeScenario(
            "device-lost", 1, identity, versions, status_all("device_lost", "NVML_ERROR_GPU_IS_LOST")
        ),
        "no-device": FakeScenario("no-device", 0, empty_identity(), versions, {}),
        "valid-zero": FakeScenario(
            "valid-zero", 1, identity, versions, ok_metrics(util_gpu=0.0, power=15.0)
        ),
    }


class FakeNvmlBackend:
    """Deterministic NVML stand-in for CPU CI. One scenario per instance."""

    def __init__(self, scenario: str) -> None:
        table = scenario_table()
        require(scenario in table, f"unknown fake scenario {scenario!r}")
        self.spec = table[scenario]

    def gpu_count(self) -> int:
        if self.spec.count_error is not None:
            raise self.spec.count_error
        return self.spec.count

    def identity(self, index: int) -> GpuIdentity:
        self._require_index(index)
        return self.spec.identity

    def versions(self) -> ToolVersions:
        return self.spec.versions

    def read_numeric(self, index: int, metric: str) -> int | float:
        fake = self._metric(index, metric)
        if fake.status != "ok":
            raise ProbeFailure(fake.status, fake.detail)
        require(fake.value is not None, f"{metric} ok fake missing value")
        return fake.value

    def read_throttle(self, index: int) -> list[str]:
        fake = self._metric(index, THROTTLE_METRIC)
        if fake.status != "ok":
            raise ProbeFailure(fake.status, fake.detail)
        return list(fake.reasons or ())

    def _require_index(self, index: int) -> None:
        if self.spec.count == 0:
            raise ProbeFailure("unsupported", "no device")
        if index < 0 or index >= self.spec.count:
            raise ProbeFailure("device_lost", "index out of range")

    def _metric(self, index: int, name: str) -> FakeMetric:
        self._require_index(index)
        fake = self.spec.metrics.get(name)
        require(fake is not None, f"scenario {self.spec.name} missing metric {name}")
        return fake


def discover_scenario(name: str, agoge_run_id: str | None = None) -> dict[str, Any]:
    run_id = agoge_run_id if agoge_run_id is not None else f"cap_fixture_{name}"
    return discover(
        FakeNvmlBackend(name),
        agoge_run_id=run_id,
        hostname=SCENARIO_ENVELOPE["hostname"],
        timestamp_utc=SCENARIO_ENVELOPE["timestamp_utc"],
        monotonic_ns=SCENARIO_ENVELOPE["monotonic_ns"],
    )


def correlation_capability_snapshot() -> dict[str, Any]:
    """Capability record bound to the committed F0 correlation fixture run."""
    return discover_scenario("partial-support", agoge_run_id="run_minicpm5_fixture_001")


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


def check_capability_time(rec: dict[str, Any]) -> None:
    require_nonempty_str(rec.get("timestamp_utc"), "timestamp_utc required")
    require(str(rec.get("timestamp_utc", "")).endswith("Z"), "timestamp_utc must end in Z")
    require_nonneg_int(rec.get("monotonic_ns"), "monotonic_ns must be a non-negative int")


def check_capability_collector(rec: dict[str, Any]) -> None:
    collector = rec.get("collector")
    require(isinstance(collector, dict), "collector object required")
    require_nonempty_str(collector.get("id"), "collector.id required")
    require_nonempty_str(collector.get("version"), "collector.version required")


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
