#!/usr/bin/env python3
"""Bounded per-run GPU sampler for one Agoge training invocation.

Start sampling, run the trainer yourself, then stop. The process is tied to
one ``agoge_run_id`` and exits on stop or ``--max-duration-s``. It does not
launch Agoge, infer training phase, or stay installed as a service.

CPU tests pass ``--fake`` (see ``nvml_fakes.py``). Live NVML is the RTX 5080
recipe in ``recipes/f0-gpu-sampler.md`` and is not what ``ci-cpu`` runs.

Usage:
    python3 tools/f0_gpu_sampler.py start --run-id <id>
    python3 tools/f0_gpu_sampler.py stop --run-id <id>
    python3 tools/f0_gpu_sampler.py run --run-id <id> -- <command...>
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import socket
import statistics
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import f0_clock
from f0_gpu_sample import (
    COLLECTOR_ID,
    COLLECTOR_VERSION,
    build_gpu_sample,
    copy_gpu,
    read_performance_state,
    read_physical,
    read_throttle_sample,
)
from f0_measurements import SchemaError
from host_cli import execute, spawn
from nvml_capability import NvmlBackend, discover
from nvml_fakes import FakeNvmlBackend
from nvml_live import LiveNvmlBackend

# The floor lives in f0_efficiency_numbers. Importing that module pulls the
# efficiency stack into the sampler; the literal here is checked against it
# in tools/check_f0_gpu_sampler.py.
HEADROOM_FLOOR_MIB = 2048
DEFAULT_CADENCE_MS = 1000
DEFAULT_MAX_DURATION_S = 7200.0
COLLECT_BUDGET_NS = 50_000_000
OVERHEAD_RATIO_NUM = 5
OVERHEAD_RATIO_DEN = 100
NS_PER_S = 1_000_000_000
SESSION_SCHEMA = "bkl.f0_sampler_session.v1"
OVERHEAD_SCHEMA = "bkl.f0_sampler_overhead.v1"
REPO_ROOT = Path(__file__).resolve().parents[1]


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
    """Host clocks. One boot, one machine; not a distributed time source."""

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


@dataclass
class ArmResult:
    directory: Path
    snapshot: dict[str, Any]
    start: dict[str, Any]
    refused: str | None
    code: int


@dataclass
class LoopState:
    backend: NvmlBackend
    snapshot: dict[str, Any]
    clock: Clock
    cadence_ms: int
    cadence_ns: int
    origin_ns: int
    max_duration_ns: int
    max_samples: int | None
    gpu_index: int
    samples_path: Path
    stop_path: Path
    stop_event: threading.Event | None
    seed_count: int = 0
    previous_ns: int | None = None


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
            item = json.loads(line)
            if isinstance(item, dict):
                rows.append(item)
    return rows


def ensure_new_run(directory: Path) -> None:
    if directory.exists():
        raise SamplerError(f"run directory already exists: {directory}")
    directory.mkdir(parents=True)


def open_backend(fake: str | None, injected: NvmlBackend | None) -> NvmlBackend:
    if injected is not None:
        return injected
    if fake is not None:
        return FakeNvmlBackend(fake)
    return LiveNvmlBackend()


def close_backend(backend: NvmlBackend) -> None:
    close = getattr(backend, "close", None)
    if callable(close):
        close()


def metric_ok_int(metrics: dict[str, Any], name: str) -> int | None:
    record = metrics.get(name)
    if not isinstance(record, dict) or record.get("status") != "ok":
        return None
    value = record.get("value")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def free_mib(snapshot: dict[str, Any]) -> tuple[str, int | None]:
    metrics = snapshot["metrics"]
    for name in ("headroom", "vram_free"):
        value = metric_ok_int(metrics, name)
        if value is not None:
            return "ok", value
    return str(metrics["headroom"]["status"]), None


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


def arm(opts: SampleOptions, backend: NvmlBackend, clock: Clock, directory: Path) -> ArmResult:
    start = observation_from(clock, "start")
    hostname = opts.hostname if opts.hostname else socket.gethostname()
    snapshot = discover(
        backend,
        agoge_run_id=opts.run_id,
        hostname=hostname,
        timestamp_utc=start["timestamp_utc"],
        monotonic_ns=start["monotonic_ns"],
        gpu_index=opts.gpu_index,
    )
    write_json(run_paths(directory)["capability"], snapshot)
    reason = refusal_reason(snapshot, opts)
    if reason is None:
        return ArmResult(directory, snapshot, start, None, 0)
    write_refusal(directory, snapshot, reason)
    return ArmResult(directory, snapshot, start, reason, refusal_code(reason))


def stopped(state: LoopState) -> bool:
    if state.stop_event is not None and state.stop_event.is_set():
        return True
    return state.stop_path.is_file()


def past_duration(state: LoopState) -> bool:
    return state.clock.monotonic_ns() - state.origin_ns > state.max_duration_ns


def hit_max(count: int, max_samples: int | None) -> bool:
    return max_samples is not None and count >= max_samples


def sleep_until(clock: Clock, deadline_ns: int) -> None:
    delay = deadline_ns - clock.monotonic_ns()
    if delay > 0:
        clock.sleep_ns(delay)


def begin_reading(backend: NvmlBackend) -> None:
    reset = getattr(backend, "begin_reading", None)
    if callable(reset):
        reset()


def capture_sample(state: LoopState) -> dict[str, Any]:
    begin_reading(state.backend)
    wall_ns = state.clock.wall_ns()
    mono = state.clock.monotonic_ns()
    physical = read_physical(state.backend, state.gpu_index)
    throttle = read_throttle_sample(state.backend, state.gpu_index)
    performance = read_performance_state(state.backend, state.gpu_index)
    duration = state.clock.monotonic_ns() - mono
    return build_gpu_sample(
        state.snapshot,
        cadence_ms=state.cadence_ms,
        timestamp_utc=f0_clock.ns_to_rfc3339_utc(wall_ns),
        monotonic_ns=mono,
        physical=physical,
        throttle=throttle,
        performance_state=performance,
        collect_duration_ns=duration,
        previous_ns=state.previous_ns,
    )


def loop_should_stop(state: LoopState, count: int) -> bool:
    if stopped(state):
        return True
    if past_duration(state):
        return True
    return hit_max(count, state.max_samples)


def sample_loop(state: LoopState) -> int:
    count = state.seed_count
    while not loop_should_stop(state, count):
        if count > 0:
            sleep_until(state.clock, state.origin_ns + count * state.cadence_ns)
        if loop_should_stop(state, count):
            break
        sample = capture_sample(state)
        append_jsonl(state.samples_path, sample)
        state.previous_ns = sample["monotonic_ns"]
        count += 1
    return count


def finalize(armed: ArmResult, clock: Clock, cadence_ms: int) -> None:
    files = run_paths(armed.directory)
    end = observation_from(clock, "end")
    samples = read_jsonl(files["samples"])
    write_json(files["skew"], make_skew(armed.snapshot, armed.start, end, clock))
    write_json(files["overhead"], make_overhead(armed.snapshot["agoge_run_id"], cadence_ms, samples))


def validate_options(opts: SampleOptions) -> None:
    check_run_id(opts.run_id)
    check_cadence(opts.cadence_ms)
    duration_ns(opts.max_duration_s)
    check_max_samples(opts.max_samples)
    check_gpu_index(opts.gpu_index)


def sample_with(opts: SampleOptions, backend: NvmlBackend, clock: Clock) -> int:
    validate_options(opts)
    directory = opts.out_root / opts.run_id
    ensure_new_run(directory)
    armed = arm(opts, backend, clock, directory)
    write_session(armed, opts, pid=None, status="refused" if armed.refused else "running")
    if armed.refused:
        return armed.code
    state = LoopState(
        backend=backend,
        snapshot=armed.snapshot,
        clock=clock,
        cadence_ms=opts.cadence_ms,
        cadence_ns=opts.cadence_ms * 1_000_000,
        origin_ns=armed.start["monotonic_ns"],
        max_duration_ns=duration_ns(opts.max_duration_s),
        max_samples=opts.max_samples,
        gpu_index=opts.gpu_index,
        samples_path=run_paths(directory)["samples"],
        stop_path=run_paths(directory)["stop"],
        stop_event=None,
    )
    sample_loop(state)
    finalize(armed, clock, opts.cadence_ms)
    session = read_json(run_paths(directory)["session"])
    session["status"] = "stopped"
    write_json(run_paths(directory)["session"], session)
    return 0


def sample_run(opts: SampleOptions, *, backend: NvmlBackend | None = None, clock: Clock | None = None) -> int:
    owns_backend = backend is None
    active = open_backend(opts.fake, backend)
    active_clock = clock if clock is not None else LiveClock()
    try:
        return sample_with(opts, active, active_clock)
    finally:
        if owns_backend:
            close_backend(active)


def session_body(armed: ArmResult, opts: SampleOptions, pid: int | None, status: str) -> dict[str, Any]:
    files = run_paths(armed.directory)
    return {
        "schema_version": SESSION_SCHEMA,
        "agoge_run_id": opts.run_id,
        "status": status,
        "pid": pid,
        "cadence_ms": opts.cadence_ms,
        "max_duration_s": opts.max_duration_s,
        "max_samples": opts.max_samples,
        "gpu_index": opts.gpu_index,
        "fake": opts.fake,
        "hostname": armed.snapshot["host"]["hostname"],
        "origin_ns": armed.start["monotonic_ns"],
        "start_observation": armed.start,
        "collector": {"id": COLLECTOR_ID, "version": COLLECTOR_VERSION},
        "samples_path": str(files["samples"].resolve()),
        "stop_path": str(files["stop"].resolve()),
        "capability_path": str(files["capability"].resolve()),
        "skew_path": str(files["skew"].resolve()),
        "overhead_path": str(files["overhead"].resolve()),
        "refused": armed.refused,
    }


def write_session(armed: ArmResult, opts: SampleOptions, pid: int | None, status: str) -> dict[str, Any]:
    body = session_body(armed, opts, pid, status)
    write_json(run_paths(armed.directory)["session"], body)
    return body


def spawn_loop(session_path: Path) -> Any:
    return spawn(
        [sys.executable, str(Path(__file__).resolve()), "loop", "--session", str(session_path)],
        cwd=REPO_ROOT,
    )


def start_session(opts: SampleOptions) -> int:
    validate_options(opts)
    directory = opts.out_root / opts.run_id
    ensure_new_run(directory)
    backend = open_backend(opts.fake, None)
    clock = LiveClock()
    try:
        armed = arm(opts, backend, clock, directory)
    finally:
        close_backend(backend)
    if armed.refused:
        write_session(armed, opts, pid=None, status="refused")
        print(armed.refused, file=sys.stderr)
        return armed.code
    session_path = run_paths(directory)["session"]
    write_session(armed, opts, pid=None, status="running")
    process = spawn_loop(session_path)
    session = read_json(session_path)
    session["pid"] = process.pid
    write_json(session_path, session)
    print(f"sampler pid {process.pid} dir {directory}")
    return 0


def process_state(pid: int) -> str | None:
    try:
        text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return None
    end = text.rfind(")")
    fields = text[end + 2 :].split()
    if end < 0 or not fields:
        return None
    return fields[0]


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
    cmdline_path = Path(f"/proc/{pid}/cmdline")
    try:
        text = cmdline_path.read_bytes().replace(b"\x00", b" ").decode("utf-8", "replace")
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


def request_exit(session: dict[str, Any]) -> None:
    Path(str(session["stop_path"])).write_text("stop\n", encoding="utf-8")
    pid = session.get("pid")
    if not isinstance(pid, int) or isinstance(pid, bool):
        return
    if sampler_pid(pid):
        escalate_stop(pid, stop_timeout(session))
    reap(pid)


def finalize_session(session: dict[str, Any], directory: Path) -> None:
    if run_paths(directory)["skew"].is_file():
        return
    snapshot = read_json(Path(str(session["capability_path"])))
    start = session["start_observation"]
    clock = LiveClock()
    end = observation_from(clock, "end")
    samples = read_jsonl(Path(str(session["samples_path"])))
    write_json(Path(str(session["skew_path"])), make_skew(snapshot, start, end, clock))
    cadence = session["cadence_ms"] if isinstance(session.get("cadence_ms"), int) else DEFAULT_CADENCE_MS
    write_json(Path(str(session["overhead_path"])), make_overhead(str(session["agoge_run_id"]), cadence, samples))


def stop_session(run_id: str, out_root: Path) -> int:
    directory = out_root / check_run_id(run_id)
    session_path = run_paths(directory)["session"]
    if not session_path.is_file():
        raise SamplerError(f"no sampler session at {session_path}")
    session = read_json(session_path)
    if session.get("status") == "refused":
        return 0
    request_exit(session)
    if session.get("status") != "refused":
        finalize_session(session, directory)
    session["status"] = "stopped"
    session["pid"] = None
    write_json(session_path, session)
    print(f"sampler stopped {directory}")
    return 0


def install_stop_on_term(event: threading.Event) -> None:
    def handle(_signum: int, _frame: Any) -> None:
        event.set()

    signal.signal(signal.SIGTERM, handle)
    signal.signal(signal.SIGINT, handle)


def loop_main(session_path: Path) -> int:
    session = read_json(session_path)
    event = threading.Event()
    install_stop_on_term(event)
    backend = open_backend(session.get("fake"), None)
    clock = LiveClock()
    snapshot = read_json(Path(str(session["capability_path"])))
    state = LoopState(
        backend=backend,
        snapshot=snapshot,
        clock=clock,
        cadence_ms=int(session["cadence_ms"]),
        cadence_ns=int(session["cadence_ms"]) * 1_000_000,
        origin_ns=int(session["origin_ns"]),
        max_duration_ns=duration_ns(float(session["max_duration_s"])),
        max_samples=session.get("max_samples"),
        gpu_index=int(session["gpu_index"]),
        samples_path=Path(str(session["samples_path"])),
        stop_path=Path(str(session["stop_path"])),
        stop_event=event,
    )
    try:
        sample_loop(state)
    finally:
        close_backend(backend)
    return 0


def take_first_sample(state: LoopState) -> None:
    sample = capture_sample(state)
    append_jsonl(state.samples_path, sample)
    state.previous_ns = int(sample["monotonic_ns"])
    state.seed_count = 1


def run_argv(argv: list[str]) -> int:
    try:
        completed = execute(argv)
    except OSError as error:
        print(error, file=sys.stderr)
        return 127
    return int(completed.returncode)


def join_timeout(opts: SampleOptions) -> float:
    return max(5.0, opts.cadence_ms / 1000 + 2.0)


def run_bracket(opts: SampleOptions, argv: list[str]) -> int:
    if not argv:
        raise SamplerError("run requires a command after --")
    validate_options(opts)
    directory = opts.out_root / opts.run_id
    ensure_new_run(directory)
    backend = open_backend(opts.fake, None)
    clock = LiveClock()
    event = threading.Event()
    thread: threading.Thread | None = None
    code = 1
    try:
        armed = arm(opts, backend, clock, directory)
        write_session(armed, opts, pid=None, status="refused" if armed.refused else "running")
        if armed.refused:
            print(armed.refused, file=sys.stderr)
            return armed.code
        state = LoopState(
            backend=backend,
            snapshot=armed.snapshot,
            clock=clock,
            cadence_ms=opts.cadence_ms,
            cadence_ns=opts.cadence_ms * 1_000_000,
            origin_ns=armed.start["monotonic_ns"],
            max_duration_ns=duration_ns(opts.max_duration_s),
            max_samples=opts.max_samples,
            gpu_index=opts.gpu_index,
            samples_path=run_paths(directory)["samples"],
            stop_path=run_paths(directory)["stop"],
            stop_event=event,
        )
        take_first_sample(state)
        thread = threading.Thread(target=sample_loop, args=(state,), name="bkl-gpu-sampler", daemon=False)
        thread.start()
        code = run_argv(argv)
        return code
    finally:
        event.set()
        if thread is not None:
            thread.join(timeout=join_timeout(opts))
        close_backend(backend)
        session_path = run_paths(directory)["session"]
        if session_path.is_file():
            session = read_json(session_path)
            if session.get("status") != "refused":
                snapshot = read_json(run_paths(directory)["capability"])
                start = session["start_observation"]
                armed_done = ArmResult(directory, snapshot, start, None, 0)
                finalize(armed_done, clock, opts.cadence_ms)
                session["status"] = "stopped"
                session["pid"] = None
                write_json(session_path, session)


def command_argv(raw: list[str] | None) -> list[str]:
    args = list(raw or [])
    if args and args[0] == "--":
        args = args[1:]
    return args


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out", type=Path, default=Path("results/f0-sampler"))
    parser.add_argument("--cadence-ms", type=int, default=DEFAULT_CADENCE_MS)
    parser.add_argument("--max-duration-s", type=float, default=DEFAULT_MAX_DURATION_S)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--hostname", default=None)
    parser.add_argument("--fake", default=None)
    parser.add_argument("--allow-low-headroom", action="store_true")
    parser.add_argument("--allow-other-compute-capability", action="store_true")


def options_from_args(args: argparse.Namespace) -> SampleOptions:
    required_cc = None if args.allow_other_compute_capability else "12.0"
    return SampleOptions(
        run_id=args.run_id,
        out_root=args.out,
        cadence_ms=args.cadence_ms,
        max_duration_s=args.max_duration_s,
        max_samples=args.max_samples,
        gpu_index=args.gpu_index,
        hostname=args.hostname,
        fake=args.fake,
        require_headroom=not args.allow_low_headroom,
        require_compute_capability=required_cc,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    start = sub.add_parser("start", help="Discover capability and sample until stop or the duration bound")
    add_common(start)
    stop = sub.add_parser("stop", help="Stop the sampler for this run id and write clock skew")
    stop.add_argument("--run-id", required=True)
    stop.add_argument("--out", type=Path, default=Path("results/f0-sampler"))
    run = sub.add_parser("run", help="Sample around a command and stop when it exits")
    add_common(run)
    run.add_argument("argv", nargs=argparse.REMAINDER)
    loop = sub.add_parser("loop", help=argparse.SUPPRESS)
    loop.add_argument("--session", type=Path, required=True)
    return parser


def dispatch(args: argparse.Namespace) -> int:
    if args.command == "start":
        return start_session(options_from_args(args))
    if args.command == "stop":
        return stop_session(args.run_id, args.out)
    if args.command == "run":
        return run_bracket(options_from_args(args), command_argv(args.argv))
    if args.command == "loop":
        return loop_main(args.session)
    raise SamplerError(f"unknown command {args.command}")


def main(argv: list[str]) -> int:
    try:
        return dispatch(build_parser().parse_args(argv[1:]))
    except (
        SamplerError,
        SchemaError,
        f0_clock.ClockError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        print(error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
