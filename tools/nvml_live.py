"""Optional live NVML backend via ctypes. Not imported by CPU CI self-tests.

Used by the RTX 5080 evidence recipe. Does not estimate unsupported sensors
and does not sample in a loop (#53).
"""

from __future__ import annotations

import ctypes
import os
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
REQUIRED_SYMBOLS = (
    "nvmlInit_v2",
    "nvmlShutdown",
    "nvmlDeviceGetCount_v2",
    "nvmlDeviceGetHandleByIndex_v2",
)


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


class LiveNvmlBackend:
    """ctypes NVML backend for the RTX 5080 evidence recipe."""

    def __init__(self, library: str = LIBRARY_DEFAULT) -> None:
        self._lib_name = library
        self._lib: Any | None = None
        self._init_error: ProbeFailure | None = None
        self._vram: MemoryInfo | None = None
        self._vram_error: ProbeFailure | None = None
        try:
            self._lib = ctypes.CDLL(library)
            _bind_required(self._lib)
        except (OSError, AttributeError) as error:
            self._init_error = ProbeFailure("unsupported", f"NVML library not usable: {error}")
            self._lib = None
            return
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
        return GpuIdentity(
            pci_bus_id=_pci_bus_id(lib, handle),
            uuid=_named_string(lib, "nvmlDeviceGetUUID", handle, 80),
            name=_named_string(lib, "nvmlDeviceGetName", handle, 96),
            compute_capability=_compute_capability(lib, handle),
        )

    def versions(self) -> ToolVersions:
        lib = self._require_lib()
        driver = _named_string(lib, "nvmlSystemGetDriverVersion", None, 80)
        nvml = _named_string(lib, "nvmlSystemGetNVMLVersion", None, 80)
        return ToolVersions(nvml_version=nvml, driver_version=driver, nvidia_smi=None)

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
            "vram_used": lambda h: self._vram_mib(h, "used"),
            "vram_free": lambda h: self._vram_mib(h, "free"),
            "vram_total": lambda h: self._vram_mib(h, "total"),
            "headroom": lambda h: self._vram_mib(h, "free"),
            PSTATE_METRIC: lambda h: _pstate(lib, h),
        }
        reader = readers.get(metric)
        require(reader is not None, f"unknown metric {metric}")
        return reader(handle)

    def read_throttle(self, index: int) -> list[str]:
        handle = self._handle(index)
        lib = self._require_lib()
        mask = ctypes.c_ulonglong()
        _invoke(lib, "nvmlDeviceGetCurrentClocksThrottleReasons", handle, ctypes.byref(mask), what="throttle")
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

    def _vram_mib(self, handle: Any, which: str) -> int:
        info = self._vram_info(handle)
        mapping = {"total": info.total, "free": info.free, "used": info.used}
        return int(int(mapping[which]) // (1024 * 1024))

    def _vram_info(self, handle: Any) -> MemoryInfo:
        if self._vram is not None:
            return self._vram
        if self._vram_error is not None:
            raise self._vram_error
        lib = self._require_lib()
        info = MemoryInfo()
        try:
            _invoke(lib, "nvmlDeviceGetMemoryInfo", handle, ctypes.byref(info), what="vram")
        except ProbeFailure as error:
            self._vram_error = error
            raise
        self._vram = info
        return info


def _bind_required(lib: Any) -> None:
    for name in REQUIRED_SYMBOLS:
        getattr(lib, name).restype = ctypes.c_int


def _fn(lib: Any, name: str) -> Callable[..., int]:
    func = getattr(lib, name, None)
    if not callable(func):
        raise ProbeFailure("unsupported", f"{name} not exported")
    func.restype = ctypes.c_int
    return func


def _invoke(lib: Any, name: str, *args: Any, what: str) -> None:
    fn = _fn(lib, name)
    _check(fn(*args), what)


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


def _named_string(lib: Any, name: str, handle: Any, size: int) -> str | None:
    try:
        fn = _fn(lib, name)
    except ProbeFailure:
        return None
    if handle is None:
        return _nvml_string(lambda buf, n: fn(buf, n), size)
    return _nvml_string(lambda buf, n: fn(handle, buf, n), size)


def _compute_capability(lib: Any, handle: Any) -> str | None:
    try:
        fn = _fn(lib, "nvmlDeviceGetCudaComputeCapability")
    except ProbeFailure:
        return None
    major = ctypes.c_int()
    minor = ctypes.c_int()
    code = int(fn(handle, ctypes.byref(major), ctypes.byref(minor)))
    if code != NVML_SUCCESS:
        return None
    return f"{int(major.value)}.{int(minor.value)}"


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
    _invoke(lib, "nvmlDeviceGetPowerUsage", handle, ctypes.byref(milliwatts), what="power")
    return int(milliwatts.value) / 1000.0


def _temp_gpu(lib: Any, handle: Any) -> int:
    temp = ctypes.c_uint()
    _invoke(lib, "nvmlDeviceGetTemperature", handle, ctypes.c_int(0), ctypes.byref(temp), what="temp_gpu")
    return int(temp.value)


def _temp_memory(lib: Any, handle: Any) -> int:
    fn = getattr(lib, "nvmlDeviceGetMemoryTemp", None)
    if fn is None:
        raise ProbeFailure("unsupported", "nvmlDeviceGetMemoryTemp not exported")
    fn.restype = ctypes.c_int
    temp = ctypes.c_uint()
    _check(fn(handle, ctypes.byref(temp)), "temp_mem")
    return int(temp.value)


def _util(lib: Any, handle: Any, which: str) -> int:
    util = Utilization()
    _invoke(lib, "nvmlDeviceGetUtilizationRates", handle, ctypes.byref(util), what="util")
    return int(util.gpu if which == "gpu" else util.memory)


def _clock(lib: Any, handle: Any, clock_type: int) -> int:
    clock = ctypes.c_uint()
    _invoke(lib, "nvmlDeviceGetClockInfo", handle, ctypes.c_int(clock_type), ctypes.byref(clock), what="clock")
    return int(clock.value)


def _pstate(lib: Any, handle: Any) -> int:
    state = ctypes.c_int()
    _invoke(lib, "nvmlDeviceGetPerformanceState", handle, ctypes.byref(state), what="pstate")
    return int(state.value)


def discover_live(**kwargs: Any) -> dict[str, Any]:
    backend = LiveNvmlBackend(os.environ.get("BKL_NVML_LIBRARY", LIBRARY_DEFAULT))
    try:
        return discover(backend, **kwargs)
    finally:
        backend.close()
