"""Optional live NVML backend via ctypes. Not imported by CPU CI self-tests.

Used by the RTX 5080 evidence recipe. Does not estimate unsupported sensors
and does not sample in a loop (#53).
"""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
from typing import Any, Callable

from f0_measurements import PSTATE_METRIC, require
from nvml_capability import (
    GpuIdentity,
    ProbeFailure,
    ToolVersions,
    decode_throttle_reasons,
    discover,
    map_nvml_error,
)

NVML_SUCCESS = 0
LIBRARY_DEFAULT = "libnvidia-ml.so.1"


class PciInfo(ctypes.Structure):
    _fields_ = [
        ("busIdLegacy", ctypes.c_char * 16),
        ("domain", ctypes.c_uint),
        ("bus", ctypes.c_uint),
        ("device", ctypes.c_uint),
        ("pciDeviceId", ctypes.c_uint),
        ("pciSubSystemId", ctypes.c_uint),
        ("busId", ctypes.c_char * 32),
    ]


class Utilization(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


class MemoryInfo(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


def nvidia_smi_version() -> str | None:
    """Record nvidia-smi's version string if present. Not a metric source."""
    binary = shutil.which("nvidia-smi")
    if binary is None:
        return None
    try:
        proc = subprocess.run(
            [binary, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    lines = (proc.stdout or proc.stderr).strip().splitlines()
    if not lines:
        return None
    return lines[0][:120] or None


class LiveNvmlBackend:
    """ctypes NVML backend for the RTX 5080 evidence recipe."""

    def __init__(self, library: str = LIBRARY_DEFAULT) -> None:
        self._lib_name = library
        self._lib: Any | None = None
        self._init_error: ProbeFailure | None = None
        try:
            self._lib = ctypes.CDLL(library)
        except OSError as error:
            self._init_error = ProbeFailure("unsupported", f"NVML library not loadable: {error}")
            return
        _bind_nvml(self._lib)
        code = int(self._lib.nvmlInit_v2())
        if code != NVML_SUCCESS:
            self._init_error = ProbeFailure(map_nvml_error(code), f"nvmlInit_v2={code}")
            self._lib = None

    def close(self) -> None:
        if self._lib is None:
            return
        self._lib.nvmlShutdown()
        self._lib = None

    def gpu_count(self) -> int:
        lib = self._require_lib()
        count = ctypes.c_uint()
        _check(lib.nvmlDeviceGetCount_v2(ctypes.byref(count)), "count")
        return int(count.value)

    def identity(self, index: int) -> GpuIdentity:
        handle = self._handle(index)
        lib = self._require_lib()
        name = _nvml_string(lambda buf, size: lib.nvmlDeviceGetName(handle, buf, size), 96)
        uuid = _nvml_string(lambda buf, size: lib.nvmlDeviceGetUUID(handle, buf, size), 80)
        pci = _pci_bus_id(lib, handle)
        major = ctypes.c_int()
        minor = ctypes.c_int()
        cap_code = int(
            lib.nvmlDeviceGetCudaComputeCapability(handle, ctypes.byref(major), ctypes.byref(minor))
        )
        compute = f"{int(major.value)}.{int(minor.value)}" if cap_code == NVML_SUCCESS else None
        return GpuIdentity(pci_bus_id=pci, uuid=uuid, name=name, compute_capability=compute)

    def versions(self) -> ToolVersions:
        lib = self._require_lib()
        driver = _nvml_string(lambda buf, size: lib.nvmlSystemGetDriverVersion(buf, size), 80)
        nvml = _nvml_string(lambda buf, size: lib.nvmlSystemGetNVMLVersion(buf, size), 80)
        return ToolVersions(nvml_version=nvml, driver_version=driver, nvidia_smi=nvidia_smi_version())

    def read_numeric(self, index: int, metric: str) -> int | float:
        handle = self._handle(index)
        lib = self._require_lib()
        readers: dict[str, Callable[[Any], int | float]] = {
            "power": lambda h: _power_w(lib, h),
            "temperature_gpu": lambda h: _temp_gpu(lib, h),
            "temperature_memory": lambda h: _temp_memory(lib, h),
            "utilization_gpu": lambda h: _util(lib, h, "gpu"),
            "utilization_memory": lambda h: _util(lib, h, "memory"),
            "clock_graphics": lambda h: _clock(lib, h, 0),
            "clock_memory": lambda h: _clock(lib, h, 2),
            "vram_used": lambda h: _vram(lib, h, "used"),
            "vram_free": lambda h: _vram(lib, h, "free"),
            "vram_total": lambda h: _vram(lib, h, "total"),
            "headroom": lambda h: _vram(lib, h, "free"),
            PSTATE_METRIC: lambda h: _pstate(lib, h),
        }
        reader = readers.get(metric)
        require(reader is not None, f"unknown metric {metric}")
        return reader(handle)

    def read_throttle(self, index: int) -> list[str]:
        handle = self._handle(index)
        lib = self._require_lib()
        mask = ctypes.c_ulonglong()
        _check(lib.nvmlDeviceGetCurrentClocksThrottleReasons(handle, ctypes.byref(mask)), "throttle")
        return decode_throttle_reasons(int(mask.value))

    def _require_lib(self) -> Any:
        if self._init_error is not None:
            raise self._init_error
        if self._lib is None:
            raise ProbeFailure("unsupported", f"{self._lib_name} not loaded")
        return self._lib

    def _handle(self, index: int) -> Any:
        lib = self._require_lib()
        handle = ctypes.c_void_p()
        _check(lib.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(index), ctypes.byref(handle)), "handle")
        return handle


def _bind_nvml(lib: Any) -> None:
    names = (
        "nvmlInit_v2",
        "nvmlShutdown",
        "nvmlDeviceGetCount_v2",
        "nvmlDeviceGetHandleByIndex_v2",
        "nvmlDeviceGetName",
        "nvmlDeviceGetUUID",
        "nvmlDeviceGetPowerUsage",
        "nvmlDeviceGetTemperature",
        "nvmlDeviceGetUtilizationRates",
        "nvmlDeviceGetClockInfo",
        "nvmlDeviceGetMemoryInfo",
        "nvmlDeviceGetPerformanceState",
        "nvmlDeviceGetCurrentClocksThrottleReasons",
        "nvmlSystemGetDriverVersion",
        "nvmlSystemGetNVMLVersion",
        "nvmlDeviceGetCudaComputeCapability",
    )
    for name in names:
        getattr(lib, name).restype = ctypes.c_int


def _check(code: int, what: str) -> None:
    status = map_nvml_error(int(code))
    if status != "ok":
        raise ProbeFailure(status, f"{what}: nvml={int(code)}")


def _nvml_string(fn: Callable[[Any, int], int], size: int) -> str | None:
    buf = ctypes.create_string_buffer(size)
    code = int(fn(buf, size))
    if map_nvml_error(code) != "ok":
        return None
    return buf.value.decode("utf-8", "replace") or None


def _pci_bus_id(lib: Any, handle: Any) -> str | None:
    info = PciInfo()
    getter = getattr(lib, "nvmlDeviceGetPciInfo_v3", None) or getattr(lib, "nvmlDeviceGetPciInfo", None)
    if getter is None:
        return None
    if map_nvml_error(int(getter(handle, ctypes.byref(info)))) != "ok":
        return None
    bus = info.busId.decode("utf-8", "replace") or info.busIdLegacy.decode("utf-8", "replace")
    return bus or None


def _power_w(lib: Any, handle: Any) -> float:
    milliwatts = ctypes.c_uint()
    _check(lib.nvmlDeviceGetPowerUsage(handle, ctypes.byref(milliwatts)), "power")
    return int(milliwatts.value) / 1000.0


def _temp_gpu(lib: Any, handle: Any) -> int:
    temp = ctypes.c_uint()
    _check(lib.nvmlDeviceGetTemperature(handle, ctypes.c_int(0), ctypes.byref(temp)), "temp_gpu")
    return int(temp.value)


def _temp_memory(lib: Any, handle: Any) -> int:
    temp = ctypes.c_uint()
    memory_fn = getattr(lib, "nvmlDeviceGetMemoryTemp", None)
    if memory_fn is not None:
        _check(memory_fn(handle, ctypes.byref(temp)), "temp_mem")
        return int(temp.value)
    _check(lib.nvmlDeviceGetTemperature(handle, ctypes.c_int(1), ctypes.byref(temp)), "temp_mem")
    return int(temp.value)


def _util(lib: Any, handle: Any, which: str) -> int:
    util = Utilization()
    _check(lib.nvmlDeviceGetUtilizationRates(handle, ctypes.byref(util)), "util")
    return int(util.gpu if which == "gpu" else util.memory)


def _clock(lib: Any, handle: Any, clock_type: int) -> int:
    clock = ctypes.c_uint()
    _check(lib.nvmlDeviceGetClockInfo(handle, ctypes.c_int(clock_type), ctypes.byref(clock)), "clock")
    return int(clock.value)


def _vram(lib: Any, handle: Any, which: str) -> int:
    info = MemoryInfo()
    _check(lib.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(info)), "vram")
    mapping = {"total": info.total, "free": info.free, "used": info.used}
    return int(int(mapping[which]) // (1024 * 1024))


def _pstate(lib: Any, handle: Any) -> int:
    state = ctypes.c_int()
    _check(lib.nvmlDeviceGetPerformanceState(handle, ctypes.byref(state)), "pstate")
    return int(state.value)


def discover_live(**kwargs: Any) -> dict[str, Any]:
    backend = LiveNvmlBackend(os.environ.get("BKL_NVML_LIBRARY", LIBRARY_DEFAULT))
    try:
        return discover(backend, **kwargs)
    finally:
        backend.close()
