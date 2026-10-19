"""Clocks, session files, and overhead math for the per-run GPU sampler.

Kept separate so the sampler module stays under the file-length limit.
No CUDA context and no trainer process.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import pty
import signal
import statistics
import termios
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import f0_clock
from f0_gpu_sample import COLLECTOR_ID, COLLECTOR_VERSION, copy_gpu

HEADROOM_FLOOR_MIB = 2048
DEFAULT_CADENCE_MS = 1000
DEFAULT_MAX_DURATION_S = 7200.0
COLLECT_BUDGET_NS = 50_000_000
OVERHEAD_RATIO_NUM = 5
OVERHEAD_RATIO_DEN = 100
NS_PER_S = 1_000_000_000
SESSION_SCHEMA = "bkl.f0_sampler_session.v1"
OVERHEAD_SCHEMA = "bkl.f0_sampler_overhead.v1"


class SamplerError(Exception):
    """The sampler invocation cannot start or the session is unusable."""


class Clock(Protocol):
    def monotonic_ns(self) -> int:
        """CLOCK_MONOTONIC nanoseconds."""

    def wall_ns(self) -> int:
        """UTC wall time as nanoseconds since the Unix epoch."""

    def sleep_ns(self, amount: int) -> None:
        """Advance at least ``amount`` nanoseconds."""


class LiveClock:
    """Host clocks. One boot, one machine."""

    def monotonic_ns(self) -> int:
        return time.monotonic_ns()

    def wall_ns(self) -> int:
        return time.time_ns()

    def sleep_ns(self, amount: int) -> None:
        if amount > 0:
            time.sleep(amount / NS_PER_S)


class ScriptedClock:
    """Deterministic clock for CPU tests. Sleep advances both timelines equally."""

    def __init__(self, wall_ns: int, monotonic_ns: int) -> None:
        self._wall = wall_ns
        self._mono = monotonic_ns

    def monotonic_ns(self) -> int:
        return self._mono

    def wall_ns(self) -> int:
        return self._wall

    def sleep_ns(self, amount: int) -> None:
        if amount > 0:
            self._wall += amount
            self._mono += amount


@dataclass
class SampleOptions:
    run_id: str
    out_root: Path
    cadence_ms: int = DEFAULT_CADENCE_MS
    max_duration_s: float = DEFAULT_MAX_DURATION_S
    max_samples: int | None = None
    gpu_index: int = 0
    hostname: str | None = None
    fake: str | None = None
    require_headroom: bool = True
    require_compute_capability: str | None = "12.0"


def observation_from(clock: Clock, role: str) -> dict[str, Any]:
    wall_ns = clock.wall_ns()
    return {
        "role": role,
        "timestamp_utc": f0_clock.ns_to_rfc3339_utc(wall_ns),
        "monotonic_ns": clock.monotonic_ns(),
        "wall_ns": wall_ns,
    }


def run_id_charset(run_id: str) -> bool:
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-")
    if len(run_id) > 128 or run_id[0] not in allowed:
        return False
    return all(char in allowed for char in run_id)


def check_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or not run_id:
        raise SamplerError("agoge_run_id is required")
    if run_id in {".", ".."} or not run_id_charset(run_id):
        raise SamplerError("agoge_run_id must be 1–128 characters from [A-Za-z0-9._-]")
    return run_id


def check_cadence(cadence_ms: int) -> int:
    if isinstance(cadence_ms, bool) or not isinstance(cadence_ms, int) or cadence_ms <= 0:
        raise SamplerError("cadence_ms must be a positive int")
    return cadence_ms


def duration_ns(seconds: float) -> int:
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
        raise SamplerError("max_duration_s must be a positive number")
    if not math.isfinite(float(seconds)) or float(seconds) <= 0:
        raise SamplerError("max_duration_s must be a positive number")
    return int(float(seconds) * NS_PER_S)


def check_max_samples(value: int | None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SamplerError("max_samples must be a positive int")


def check_gpu_index(index: int) -> None:
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise SamplerError("gpu_index must be a non-negative int")


def run_paths(directory: Path) -> dict[str, Path]:
    return {
        "capability": directory / "capability.json",
        "samples": directory / "samples.jsonl",
        "stop": directory / "STOP",
        "session": directory / "session.json",
        "skew": directory / "clock-skew.json",
        "overhead": directory / "overhead.json",
        "refusal": directory / "refusal.json",
    }


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(obj, separators=(",", ":"), sort_keys=True) + "\n")
        handle.flush()


def read_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise SamplerError(f"{path} must contain a JSON object")
    return obj


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def ensure_new_run(directory: Path) -> None:
    if directory.exists():
        raise SamplerError(f"run directory already exists: {directory}")
    directory.mkdir(parents=True)


def metric_ok_int(metrics: dict[str, Any], name: str) -> int | None:
    record = metrics.get(name)
    if not isinstance(record, dict) or record.get("status") != "ok":
        return None
    value = record.get("value")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def metric_status(metrics: dict[str, Any], name: str) -> str:
    record = metrics.get(name)
    if isinstance(record, dict) and isinstance(record.get("status"), str):
        return str(record["status"])
    return "unavailable"


def free_mib(snapshot: dict[str, Any]) -> tuple[str, int | None]:
    metrics = snapshot.get("metrics")
    if not isinstance(metrics, dict):
        return "unavailable", None
    for name in ("headroom", "vram_free"):
        value = metric_ok_int(metrics, name)
        if value is not None:
            return "ok", value
    return metric_status(metrics, "headroom"), None


def identity_problem(snapshot: dict[str, Any]) -> str | None:
    gpu = snapshot.get("gpu")
    if not isinstance(gpu, dict):
        return "no gpu identity"
    if gpu.get("uuid") or gpu.get("pci_bus_id"):
        return None
    return "no gpu identity"


def cc_problem(snapshot: dict[str, Any], required: str | None) -> str | None:
    if required is None:
        return None
    actual = snapshot["gpu"].get("compute_capability")
    if actual == required:
        return None
    return f"compute_capability {actual} != {required}"


def headroom_problem(snapshot: dict[str, Any]) -> str | None:
    status, value = free_mib(snapshot)
    if status != "ok" or value is None:
        return f"headroom {status}"
    if value < HEADROOM_FLOOR_MIB:
        return f"headroom {value} MiB below {HEADROOM_FLOOR_MIB} MiB"
    return None


def refusal_reason(snapshot: dict[str, Any], opts: SampleOptions) -> str | None:
    identity = identity_problem(snapshot)
    if identity is not None:
        return identity
    capability = cc_problem(snapshot, opts.require_compute_capability)
    if capability is not None:
        return capability
    if opts.require_headroom:
        return headroom_problem(snapshot)
    return None


def refusal_code(reason: str) -> int:
    if reason.startswith("headroom"):
        return 3
    return 2


def clock_resolution_for(clock: Clock) -> dict[str, int]:
    if isinstance(clock, LiveClock):
        return f0_clock.read_clock_resolution()
    return {"monotonic_ns": 1, "realtime_ns": 1}


def make_skew(snapshot: dict[str, Any], start: dict[str, Any], end: dict[str, Any], clock: Clock) -> dict[str, Any]:
    report = f0_clock.calibrate([start, end], clock_resolution=clock_resolution_for(clock))
    return f0_clock.build_clock_skew_record(
        report,
        agoge_run_id=snapshot["agoge_run_id"],
        host={"hostname": snapshot["host"]["hostname"]},
        gpu=copy_gpu(snapshot["gpu"]),
        collector={"id": COLLECTOR_ID, "version": COLLECTOR_VERSION},
    )


def _ints(samples: list[dict[str, Any]], key: str) -> list[int]:
    values: list[int] = []
    for sample in samples:
        value = sample.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            values.append(value)
    return values


def median_number(values: list[int]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def sample_headroom_mib(sample: dict[str, Any]) -> int | None:
    for name in ("headroom", "vram_free"):
        value = metric_ok_int(sample, name)
        if value is not None:
            return value
    return None


def headroom_readings(samples: list[dict[str, Any]]) -> list[int]:
    readings: list[int] = []
    for sample in samples:
        value = sample_headroom_mib(sample)
        if value is not None:
            readings.append(value)
    return readings


def count_below_floor(readings: list[int]) -> int:
    return sum(1 for value in readings if value < HEADROOM_FLOOR_MIB)


def floor_held(readings: list[int]) -> bool | None:
    if not readings:
        return None
    return min(readings) >= HEADROOM_FLOOR_MIB


def headroom_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    readings = headroom_readings(samples)
    return {
        "headroom_floor_mib": HEADROOM_FLOOR_MIB,
        "samples_below_floor": count_below_floor(readings),
        "min_headroom_mib": min(readings) if readings else None,
        "floor_held": floor_held(readings),
    }


def overhead_within_budget(median_collect: float | None, cadence_ms: int) -> bool | None:
    if median_collect is None:
        return None
    if median_collect > COLLECT_BUDGET_NS:
        return False
    limit = cadence_ms * 1_000_000 * OVERHEAD_RATIO_NUM // OVERHEAD_RATIO_DEN
    return median_collect <= limit


def overhead_ratio(median_collect: float | None, cadence_ms: int) -> float | None:
    if median_collect is None or cadence_ms <= 0:
        return None
    return median_collect / (cadence_ms * 1_000_000)


def make_overhead(run_id: str, cadence_ms: int, samples: list[dict[str, Any]]) -> dict[str, Any]:
    collect = _ints(samples, "collect_duration_ns")
    intervals = _ints(samples, "observed_interval_ns")
    jitter = _ints(samples, "jitter_ns")
    median_collect = median_number(collect)
    report = {
        "schema_version": OVERHEAD_SCHEMA,
        "record_kind": "bkl_sampler_overhead",
        "agoge_run_id": run_id,
        "collector": {"id": COLLECTOR_ID, "version": COLLECTOR_VERSION},
        "cadence_ms": cadence_ms,
        "sample_count": len(samples),
        "collect_duration_ns": {"median": median_collect, "max": max(collect) if collect else None},
        "observed_interval_ns": {"median": median_number(intervals), "max": max(intervals) if intervals else None},
        "jitter_ns": {"median": median_number(jitter), "max": max(jitter) if jitter else None},
        "overhead_ratio": overhead_ratio(median_collect, cadence_ms),
        "overhead_budget_ratio": OVERHEAD_RATIO_NUM / OVERHEAD_RATIO_DEN,
        "overhead_budget_collect_ns": COLLECT_BUDGET_NS,
        "overhead_ok": overhead_within_budget(median_collect, cadence_ms),
    }
    report.update(headroom_summary(samples))
    return report


def write_refusal(directory: Path, snapshot: dict[str, Any], reason: str) -> None:
    write_json(
        run_paths(directory)["refusal"],
        {
            "schema_version": OVERHEAD_SCHEMA,
            "record_kind": "bkl_sampler_refusal",
            "agoge_run_id": snapshot["agoge_run_id"],
            "reason": reason,
            "capability_digest": snapshot.get("capability_digest"),
            "headroom_floor_mib": HEADROOM_FLOOR_MIB,
            "collector": {"id": COLLECTOR_ID, "version": COLLECTOR_VERSION},
        },
    )


def stat_state(text: str) -> str | None:
    end = text.rfind(")")
    if end < 0:
        return None
    fields = text[end + 2 :].split()
    if not fields:
        return None
    return fields[0]


def process_state(pid: int) -> str | None:
    try:
        text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return None
    return stat_state(text)


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    state = process_state(pid)
    return state is not None and state != "Z"


def reap(pid: int) -> None:
    try:
        os.waitpid(pid, os.WNOHANG)
    except (ChildProcessError, OSError):
        return


def sampler_pid(pid: int) -> bool:
    if not pid_alive(pid):
        return False
    try:
        text = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode("utf-8", "replace")
    except OSError:
        return False
    return "f0_gpu_sampler.py" in text


def wait_until_dead(pid: int, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not pid_alive(pid):
            return True
        time.sleep(0.02)
    return not pid_alive(pid)


def stop_timeout(session: dict[str, Any]) -> float:
    cadence = session.get("cadence_ms")
    if not isinstance(cadence, int) or isinstance(cadence, bool) or cadence <= 0:
        cadence = DEFAULT_CADENCE_MS
    return max(5.0, cadence / 1000 + 2.0)


def signal_pid(pid: int, sig: int) -> None:
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        return


def escalate_stop(pid: int, timeout: float) -> None:
    if wait_until_dead(pid, timeout):
        return
    signal_pid(pid, signal.SIGTERM)
    if wait_until_dead(pid, timeout):
        return
    signal_pid(pid, signal.SIGKILL)
    wait_until_dead(pid, 2.0)


def silence_stdio() -> None:
    null_fd = os.open(os.devnull, os.O_RDWR)
    os.dup2(null_fd, 0)
    os.dup2(null_fd, 1)
    os.dup2(null_fd, 2)
    if null_fd > 2:
        os.close(null_fd)


def exec_python(command: list[str]) -> None:
    try:
        os.execvp("python3", command)  # nosec B606 — literal interpreter, not a shell
    except OSError:
        os._exit(127)


def controlling_tty() -> int | None:
    try:
        attached = os.isatty(0)
    except OSError:
        return None
    if attached:
        return 0
    return None


def ignore_stop_signals() -> tuple[Any, Any]:
    return (
        signal.signal(signal.SIGTTIN, signal.SIG_IGN),
        signal.signal(signal.SIGTTOU, signal.SIG_IGN),
    )


def restore_stop_signals(previous: tuple[Any, Any]) -> None:
    signal.signal(signal.SIGTTIN, previous[0])
    signal.signal(signal.SIGTTOU, previous[1])


def give_foreground(tty_fd: int | None, pgrp: int) -> int | None:
    if tty_fd is None:
        return None
    try:
        previous = os.tcgetpgrp(tty_fd)
        os.tcsetpgrp(tty_fd, pgrp)
    except OSError:
        return None
    return previous


def restore_foreground(tty_fd: int | None, pgrp: int | None) -> None:
    if tty_fd is None or pgrp is None:
        return
    try:
        os.tcsetpgrp(tty_fd, pgrp)
    except OSError:
        return


def run_in_foreground(command: list[str], pid_box: list[int | None], tty_fd: int | None) -> int:
    pid = os.fork()
    if pid == 0:
        os.setpgid(0, 0)
        exec_python(command)
    separate_child_group(pid)
    pid_box[0] = pid
    saved = give_foreground(tty_fd, pid)
    try:
        return wait_child(pid)
    finally:
        restore_foreground(tty_fd, saved)


def separate_child_group(pid: int) -> None:
    try:
        os.setpgid(pid, pid)
    except OSError:
        return


def forward_signal(pid: int, signum: int) -> None:
    try:
        os.killpg(pid, signum)
    except OSError:
        return


def exit_code_from_status(status: int) -> int:
    if os.WIFSIGNALED(status):
        return 128 + int(os.WTERMSIG(status))
    return int(os.WEXITSTATUS(status))


def wait_child(pid: int) -> int:
    while True:
        try:
            _child, status = os.waitpid(pid, 0)
        except InterruptedError:
            continue
        return exit_code_from_status(status)


def install_child_forwarders(pid_box: list[int | None]) -> tuple[Any, Any]:
    def handle(signum: int, _frame: Any) -> None:
        child = pid_box[0]
        if isinstance(child, int):
            forward_signal(child, signum)

    return (
        signal.signal(signal.SIGINT, handle),
        signal.signal(signal.SIGTERM, handle),
    )


def restore_signals(previous: tuple[Any, Any]) -> None:
    signal.signal(signal.SIGINT, previous[0])
    signal.signal(signal.SIGTERM, previous[1])


def enter_controlling_tty(slave: int, master: int, command: list[str]) -> None:
    os.setsid()
    os.close(master)
    try:
        fcntl.ioctl(slave, termios.TIOCSCTTY, 0)
    except OSError:
        os._exit(127)
    os.dup2(slave, 0)
    os.dup2(slave, 1)
    os.dup2(slave, 2)
    if slave > 2:
        os.close(slave)
    exec_python(command)


def wait_bounded(pid: int, seconds: float) -> int:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        child, status = os.waitpid(pid, os.WNOHANG)
        if child == pid:
            return exit_code_from_status(status)
        time.sleep(0.05)
    os.kill(pid, signal.SIGKILL)
    _child, status = os.waitpid(pid, 0)
    return exit_code_from_status(status)


def run_with_controlling_tty(command: list[str], payload: bytes, seconds: float = 3.0) -> int:
    master, slave = pty.openpty()
    pid = os.fork()
    if pid == 0:
        enter_controlling_tty(slave, master, command)
    os.close(slave)
    os.write(master, payload)
    try:
        return wait_bounded(pid, seconds)
    finally:
        os.close(master)


def request_exit(session: dict[str, Any]) -> None:
    Path(str(session["stop_path"])).write_text("stop\n", encoding="utf-8")
    pid = session.get("pid")
    if not isinstance(pid, int) or isinstance(pid, bool):
        return
    if sampler_pid(pid):
        escalate_stop(pid, stop_timeout(session))
    reap(pid)
