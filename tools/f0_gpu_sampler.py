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
import os
import signal
import socket
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import f0_clock
from f0_gpu_sample import (
    COLLECTOR_ID,
    COLLECTOR_VERSION,
    SampleMoment,
    build_gpu_sample,
    read_performance_state,
    read_physical,
    read_throttle_sample,
)
from f0_measurements import SchemaError
from f0_sampler_support import (
    COLLECT_BUDGET_NS,
    DEFAULT_CADENCE_MS,
    DEFAULT_MAX_DURATION_S,
    HEADROOM_FLOOR_MIB,
    SESSION_SCHEMA,
    Clock,
    LiveClock,
    SampleOptions,
    SamplerError,
    ScriptedClock,
    append_jsonl,
    check_cadence,
    check_gpu_index,
    check_max_samples,
    check_run_id,
    duration_ns,
    ensure_new_run,
    make_overhead,
    make_skew,
    observation_from,
    pid_alive,
    read_json,
    read_jsonl,
    refusal_code,
    refusal_reason,
    request_exit,
    run_paths,
    write_json,
    write_refusal,
)
from nvml_capability import NvmlBackend, discover
from nvml_fakes import FakeNvmlBackend
from nvml_live import LiveNvmlBackend

__all__ = [
    "COLLECT_BUDGET_NS",
    "HEADROOM_FLOOR_MIB",
    "LiveClock",
    "SampleOptions",
    "SamplerError",
    "ScriptedClock",
    "main",
    "make_overhead",
    "pid_alive",
    "read_json",
    "read_jsonl",
    "sample_run",
]


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
    return build_gpu_sample(
        state.snapshot,
        physical,
        throttle,
        performance,
        SampleMoment(
            cadence_ms=state.cadence_ms,
            timestamp_utc=f0_clock.ns_to_rfc3339_utc(wall_ns),
            monotonic_ns=mono,
            collect_duration_ns=state.clock.monotonic_ns() - mono,
            previous_ns=state.previous_ns,
        ),
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


def loop_state_for(
    armed: ArmResult,
    opts: SampleOptions,
    backend: NvmlBackend,
    clock: Clock,
    event: threading.Event | None,
) -> LoopState:
    directory = armed.directory
    return LoopState(
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


def mark_stopped(directory: Path) -> None:
    session = read_json(run_paths(directory)["session"])
    session["status"] = "stopped"
    session["pid"] = None
    write_json(run_paths(directory)["session"], session)


def sample_with(opts: SampleOptions, backend: NvmlBackend, clock: Clock) -> int:
    validate_options(opts)
    directory = opts.out_root / opts.run_id
    ensure_new_run(directory)
    armed = arm(opts, backend, clock, directory)
    write_session(armed, opts, pid=None, status="refused" if armed.refused else "running")
    if armed.refused:
        return armed.code
    sample_loop(loop_state_for(armed, opts, backend, clock, None))
    finalize(armed, clock, opts.cadence_ms)
    mark_stopped(directory)
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


def spawn_loop(session_path: Path) -> int:
    script = str(Path(__file__).resolve())
    argv = ["python3", script, "loop", "--session", str(session_path)]
    return os.posix_spawnp("python3", argv, os.environ, setsid=True)


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
    session = read_json(session_path)
    session["pid"] = spawn_loop(session_path)
    write_json(session_path, session)
    print(f"sampler pid {session['pid']} dir {directory}")
    return 0


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


def loop_from_session(session: dict[str, Any], event: threading.Event) -> LoopState:
    return LoopState(
        backend=open_backend(session.get("fake"), None),
        snapshot=read_json(Path(str(session["capability_path"]))),
        clock=LiveClock(),
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


def loop_main(session_path: Path) -> int:
    event = threading.Event()
    install_stop_on_term(event)
    state = loop_from_session(read_json(session_path), event)
    try:
        sample_loop(state)
    finally:
        close_backend(state.backend)
    return 0


def take_first_sample(state: LoopState) -> None:
    sample = capture_sample(state)
    append_jsonl(state.samples_path, sample)
    state.previous_ns = int(sample["monotonic_ns"])
    state.seed_count = 1


def run_argv(argv: list[str]) -> int:
    try:
        pid = os.posix_spawnp(argv[0], argv, os.environ)
    except OSError as error:
        print(error, file=sys.stderr)
        return 127
    _child, status = os.waitpid(pid, 0)
    return os.waitstatus_to_exitcode(status)


def join_timeout(opts: SampleOptions) -> float:
    return max(5.0, opts.cadence_ms / 1000 + 2.0)


def finish_bracket(
    directory: Path,
    clock: Clock,
    opts: SampleOptions,
    backend: NvmlBackend,
    thread: threading.Thread | None,
) -> None:
    if thread is not None:
        thread.join(timeout=join_timeout(opts))
    close_backend(backend)
    session_path = run_paths(directory)["session"]
    if not session_path.is_file():
        return
    session = read_json(session_path)
    if session.get("status") == "refused":
        return
    snapshot = read_json(run_paths(directory)["capability"])
    armed = ArmResult(directory, snapshot, session["start_observation"], None, 0)
    finalize(armed, clock, opts.cadence_ms)
    mark_stopped(directory)


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
    try:
        armed = arm(opts, backend, clock, directory)
        write_session(armed, opts, pid=None, status="refused" if armed.refused else "running")
        if armed.refused:
            print(armed.refused, file=sys.stderr)
            return armed.code
        state = loop_state_for(armed, opts, backend, clock, event)
        take_first_sample(state)
        thread = threading.Thread(target=sample_loop, args=(state,), name="bkl-gpu-sampler", daemon=False)
        thread.start()
        return run_argv(argv)
    finally:
        event.set()
        finish_bracket(directory, clock, opts, backend, thread)


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
