"""Deterministic NVML stand-ins for CPU CI (no GPU, no libnvidia-ml)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from f0_measurements import NUMERIC_METRIC_NAMES, PSTATE_METRIC, THROTTLE_METRIC, require
from nvml_capability import (
    GpuIdentity,
    ProbeFailure,
    ToolVersions,
    discover,
    empty_identity,
)

SCENARIO_ENVELOPE = {
    "timestamp_utc": "2026-09-15T04:00:00Z",
    "monotonic_ns": 0,
    "hostname": "ShipOfTheseus",
}


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


def _permission_metrics() -> dict[str, FakeMetric]:
    permission = ok_metrics(util_gpu=5.0, power=45.0)
    permission["power"] = FakeMetric("permission_denied", detail="NVML_ERROR_NO_PERMISSION")
    permission["temperature_gpu"] = FakeMetric("permission_denied", detail="NVML_ERROR_NO_PERMISSION")
    permission["temperature_memory"] = FakeMetric("permission_denied", detail="NVML_ERROR_NO_PERMISSION")
    return permission


def _transient_metrics() -> dict[str, FakeMetric]:
    transient = ok_metrics(util_gpu=20.0, power=90.0)
    transient["power"] = FakeMetric("transient_failure", detail="NVML_ERROR_TIMEOUT")
    transient["clock_graphics"] = FakeMetric("unavailable", detail="NVML_ERROR_NO_DATA")
    return transient


def scenario_table() -> dict[str, FakeScenario]:
    identity = rtx5080_identity()
    versions = fixture_tool_versions()
    partial = ok_metrics(
        util_gpu=10.0,
        power=45.0,
        temp_mem=FakeMetric("unsupported", detail="NVML_ERROR_NOT_SUPPORTED"),
    )
    return {
        "full-support": FakeScenario(
            "full-support", 1, identity, versions, ok_metrics(util_gpu=47.0, power=180.0)
        ),
        "partial-support": FakeScenario("partial-support", 1, identity, versions, partial),
        "permission-denied": FakeScenario("permission-denied", 1, identity, versions, _permission_metrics()),
        "transient-failure": FakeScenario("transient-failure", 1, identity, versions, _transient_metrics()),
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
