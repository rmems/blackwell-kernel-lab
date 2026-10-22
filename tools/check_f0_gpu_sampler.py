#!/usr/bin/env python3
"""CPU tests for the bounded F0 GPU sampler (no GPU, NVML fakes).

See docs/f0-gpu-sampler.md. Used by .github/workflows/ci-cpu.yml.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import check_f0_correlation as f0corr
import f0_clock
import f0_gpu_sampler as sampler
from f0_sampler_harness import CachingFree, StopsMidRead, bash_stop_and_fg, run_with_controlling_tty, run_writing_to_head
from f0_sampler_support import free_mib, stat_state
from f0_efficiency_numbers import HEADROOM_FLOOR_MIB, field_block
from f0_gpu_sample import COLLECTOR_ID
from f0_measurements import PHYSICAL, SchemaError
from nvml_fakes import FakeNvmlBackend
from nvml_schema import bind_samples, check_capability_snapshot

WALL0 = 1_789_408_800_000_000_000
MONO0 = 1_000_000_000


class CheckError(Exception):
    """A sampler assertion failed."""


def require(condition: object, message: str) -> None:
    if not condition:
        raise CheckError(message)


def options(tmp: Path, run_id: str, **updates: Any) -> sampler.SampleOptions:
    base = sampler.SampleOptions(
        run_id=run_id,
        out_root=tmp,
        cadence_ms=1000,
        max_duration_s=30,
        max_samples=3,
        hostname="ShipOfTheseus",
        fake="full-support",
    )
    for key, value in updates.items():
        setattr(base, key, value)
    return base


def clock() -> sampler.ScriptedClock:
    return sampler.ScriptedClock(WALL0, MONO0)


def run_dir(tmp: Path, run_id: str) -> Path:
    return tmp / run_id


def load_samples(directory: Path) -> list[dict[str, Any]]:
    return sampler.read_jsonl(directory / "samples.jsonl")


def assert_null_when_missing(sample: dict[str, Any]) -> None:
    for name, _unit in PHYSICAL:
        record = sample[name]
        if record["status"] != "ok":
            require(record["value"] is None, f"{name} missing value must be null, got {record['value']!r}")
    state = sample["performance_state"]
    if state["status"] != "ok":
        require(state["value"] is None, "performance_state missing value must be null")
    if sample["throttle"]["status"] != "ok":
        require(sample["throttle"]["reasons"] is None, "throttle reasons must be null when not ok")


def assert_sample_shape(sample: dict[str, Any], snapshot: dict[str, Any]) -> None:
    f0corr.check_sample(sample)
    require(sample["collector"]["id"] == COLLECTOR_ID, "collector id")
    require(sample["capability_digest"] == snapshot["capability_digest"], "digest stamp")
    require(sample["profile_window_ref"] is None, "sampler does not invent a profile window")
    require("phase" not in sample, "sampler must not copy Agoge phase onto a sample")
    require(sample["cuda"]["tool"] == "nvml", "cuda.tool")
    assert_null_when_missing(sample)


def write_markers(path: Path, samples: list[dict[str, Any]]) -> None:
    first, last = samples[0], samples[-1]
    rows = [
        marker_from(first, step=0, tokens=0),
        marker_from(last, step=1, tokens=128),
    ]
    text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def marker_from(sample: dict[str, Any], *, step: int, tokens: int) -> dict[str, Any]:
    return {
        "schema_version": sample["schema_version"],
        "record_kind": "agoge_marker",
        "agoge_run_id": sample["agoge_run_id"],
        "host": sample["host"],
        "gpu": sample["gpu"],
        "timestamp_utc": sample["timestamp_utc"],
        "monotonic_ns": sample["monotonic_ns"],
        "collector": {"id": "agoge-forger", "version": "fixture"},
        "cadence_ms": None,
        "model_id": "openbmb/MiniCPM4-8B",
        "model_revision": "fixture-rev",
        "dataset": {"id": "fixture-sft", "split": "train", "config_digest": None},
        "phase": "train",
        "global_step": step,
        "microstep": 0,
        "tokens_accepted": tokens,
        "examples_accepted": step,
    }


def assert_checker(directory: Path, samples: list[dict[str, Any]]) -> None:
    markers = directory / "markers.jsonl"
    write_markers(markers, samples)
    code = f0corr.main(
        [
            "check_f0_correlation.py",
            "--markers",
            str(markers),
            "--samples",
            str(directory / "samples.jsonl"),
            "--clock-skew",
            str(directory / "clock-skew.json"),
        ]
    )
    require(code == 0, "correlation checker rejected the sampler bundle")


def assert_bundle(directory: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    snapshot = json.loads((directory / "capability.json").read_text(encoding="utf-8"))
    check_capability_snapshot(snapshot)
    samples = load_samples(directory)
    require(samples, "expected samples")
    for sample in samples:
        assert_sample_shape(sample, snapshot)
    bind_samples(snapshot, samples)
    skew = json.loads((directory / "clock-skew.json").read_text(encoding="utf-8"))
    f0_clock.check_clock_skew_record(skew)
    require(skew["end_observation"] is not None, "stop writes an end observation")
    assert_checker(directory, samples)
    return snapshot, samples


def sample_scenario(tmp: Path, name: str, **updates: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    opts = options(tmp, name, fake=name, **updates)
    code = sampler.sample_run(opts, clock=clock())
    require(code == 0, f"{name} exit {code}")
    return assert_bundle(run_dir(tmp, name))


def check_stat_state_and_missing_headroom() -> None:
    require(stat_state("malformed") is None, "stat without a comm paren is not a state")
    require(stat_state("1 (python (x)) Z 0") == "Z", "state is the field after the comm")
    status, value = free_mib({"metrics": {}})
    require(status == "unavailable" and value is None, "missing headroom is unavailable, not a crash")


def check_floor_constant() -> None:
    require(sampler.HEADROOM_FLOOR_MIB == HEADROOM_FLOOR_MIB, "headroom floor drifted from efficiency")
    require(sampler.HEADROOM_FLOOR_MIB == 2048, "host floor is 2048 MiB")


def check_full_support(tmp: Path) -> None:
    _snapshot, samples = sample_scenario(tmp, "full-support")
    require(samples[0]["power"]["status"] == "ok", "full power")
    require(samples[0]["power"]["value"] == 180.0, "full power value")
    require(samples[0]["jitter_ns"] is None, "first sample has no previous interval")
    require(samples[1]["observed_interval_ns"] == 1_000_000_000, "cadence is the observed interval")
    require(samples[1]["jitter_ns"] == 0, "on-cadence jitter is a real zero")
    overhead = json.loads((run_dir(tmp, "full-support") / "overhead.json").read_text(encoding="utf-8"))
    require(overhead["overhead_ok"] is True, "fake collect stays inside the budget")
    require(overhead["floor_held"] is True, "fixture free VRAM stays above 2048 MiB")
    require(overhead["cadence_ms"] == 1000, "planned cadence recorded")
    block = field_block(samples, "power", avg_key="avg_w", peak_key="peak_w", min_key=None, unit="W")
    require(block["peak_w"] == 180.0, "efficiency peak uses the sampled watts")


def check_partial_support(tmp: Path) -> None:
    _snapshot, samples = sample_scenario(tmp, "partial-support")
    memory = samples[0]["temperature_memory"]
    require(memory["status"] == "unsupported", "partial memory temp")
    require(memory["value"] is None, "unsupported temp is null")
    require(samples[0]["power"]["status"] == "ok", "other metrics still sampled")


def check_missing_reading(tmp: Path) -> None:
    _snapshot, samples = sample_scenario(tmp, "transient-failure")
    clock_graphics = samples[0]["clock_graphics"]
    require(clock_graphics["status"] == "unavailable", "missing clock")
    require(clock_graphics["value"] is None, "missing clock is not zero")
    require(samples[0]["power"]["status"] == "transient_failure", "timeout stays transient")
    require(samples[0]["power"]["value"] is None, "timeout power is null")
    require(samples[0]["utilization_gpu"]["status"] == "ok", "util continues")


def check_permission_denied(tmp: Path) -> None:
    _snapshot, samples = sample_scenario(tmp, "permission-denied")
    power = samples[0]["power"]
    require(power["status"] == "permission_denied", "permission power")
    require(power["value"] is None, "permission must not be encoded as 0 W")
    require(samples[0]["utilization_gpu"]["status"] == "ok", "util continues")
    block = field_block(samples, "power", avg_key="avg_w", peak_key="peak_w", min_key=None, unit="W")
    require(block["peak_w"] is None, "denied power is excluded from the peak")
    require(block["invalid_count"] == len(samples), "denied readings count as invalid")


def check_device_lost(tmp: Path) -> None:
    refused = options(tmp, "device-lost-refused", fake="device-lost")
    code = sampler.sample_run(refused, clock=clock())
    require(code == 3, "lost GPU cannot prove the headroom floor")
    require(not (run_dir(tmp, "device-lost-refused") / "samples.jsonl").exists(), "refusal writes no samples")
    _snapshot, samples = sample_scenario(tmp, "device-lost", require_headroom=False)
    require(samples[0]["gpu"]["uuid"] == "GPU-fixture-0001", "lost device keeps UUID")
    require(samples[0]["power"]["status"] == "device_lost", "lost power status")
    require(samples[0]["power"]["value"] is None, "lost power is null")


def check_valid_zero(tmp: Path) -> None:
    _snapshot, samples = sample_scenario(tmp, "valid-zero")
    util = samples[0]["utilization_gpu"]
    require(util["status"] == "ok" and util["value"] == 0.0, "idle 0% stays a real zero")
    require(samples[0]["power"]["value"] == 15.0, "idle power is not a fabricated 0 W")


def check_no_device(tmp: Path) -> None:
    opts = options(tmp, "no-device", fake="no-device", max_samples=1)
    code = sampler.sample_run(opts, clock=clock())
    require(code == 2, "no device refuses to sample")
    directory = run_dir(tmp, "no-device")
    require(not (directory / "samples.jsonl").exists(), "no fabricated samples without a GPU")
    refusal = json.loads((directory / "refusal.json").read_text(encoding="utf-8"))
    require(refusal["record_kind"] == "bkl_sampler_refusal", "refusal record")
    snapshot = json.loads((directory / "capability.json").read_text(encoding="utf-8"))
    require(snapshot["device_status"] == "no_device", "capability records no device")


class LowFree:
    """Full-support fake with free VRAM under the 2048 MiB floor."""

    def __init__(self) -> None:
        self.inner = FakeNvmlBackend("full-support")

    def gpu_count(self) -> int: return self.inner.gpu_count()

    def identity(self, index: int) -> Any: return self.inner.identity(index)

    def versions(self) -> Any: return self.inner.versions()

    def read_throttle(self, index: int) -> list[str]: return self.inner.read_throttle(index)

    def read_numeric(self, index: int, metric: str) -> int | float:
        if metric in ("vram_free", "headroom"):
            return 512
        if metric == "vram_used":
            return 15791
        return self.inner.read_numeric(index, metric)


def check_low_headroom(tmp: Path) -> None:
    refused = options(tmp, "low-free", fake=None)
    code = sampler.sample_run(refused, backend=LowFree(), clock=clock())
    require(code == 3, "start refuses when free VRAM is below 2048 MiB")
    require(not (run_dir(tmp, "low-free") / "samples.jsonl").exists(), "no samples when the floor blocks start")
    allowed = options(tmp, "low-free-recorded", fake=None, require_headroom=False)
    code = sampler.sample_run(allowed, backend=LowFree(), clock=clock())
    require(code == 0, "recording a breach is allowed when the floor check is off")
    _snapshot, samples = assert_bundle(run_dir(tmp, "low-free-recorded"))
    require(samples[0]["headroom"]["value"] == 512, "breach is recorded")
    overhead = json.loads((run_dir(tmp, "low-free-recorded") / "overhead.json").read_text(encoding="utf-8"))
    require(overhead["floor_held"] is False, "overhead records the floor breach")
    require(overhead["samples_below_floor"] == len(samples), "each low sample counts")


class Costly:
    """Advances the scripted clock on every numeric read."""

    def __init__(self, scripted: sampler.ScriptedClock, cost_ns: int) -> None:
        self.inner = FakeNvmlBackend("full-support")
        self.clock = scripted
        self.cost_ns = cost_ns

    def gpu_count(self) -> int: return self.inner.gpu_count()

    def identity(self, index: int) -> Any: return self.inner.identity(index)

    def versions(self) -> Any: return self.inner.versions()

    def read_throttle(self, index: int) -> list[str]: return self.inner.read_throttle(index)

    def read_numeric(self, index: int, metric: str) -> int | float:
        self.clock.sleep_ns(self.cost_ns)
        return self.inner.read_numeric(index, metric)


def check_overhead_budget(tmp: Path) -> None:
    scripted = clock()
    opts = options(tmp, "costly", fake=None, max_samples=2)
    code = sampler.sample_run(opts, backend=Costly(scripted, 5_000_000), clock=scripted)
    require(code == 0, "over-budget collect still records samples")
    _snapshot, samples = assert_bundle(run_dir(tmp, "costly"))
    require(samples[0]["power"]["value"] == 180.0, "overhead accounting does not change power")
    require(samples[0]["collect_duration_ns"] > sampler.COLLECT_BUDGET_NS, "collect duration was measured")
    overhead = json.loads((run_dir(tmp, "costly") / "overhead.json").read_text(encoding="utf-8"))
    require(overhead["overhead_ok"] is False, "over-budget collect is flagged")
    empty = sampler.make_overhead("empty", 1000, [])
    require(empty["overhead_ratio"] is None, "no samples is not a zero overhead ratio")
    require(empty["overhead_ok"] is None, "no samples is not a passing overhead")


class LateClock(sampler.ScriptedClock):
    def sleep_ns(self, amount: int) -> None: super().sleep_ns(amount + 2_000_000)


def check_jitter_recorded(tmp: Path) -> None:
    opts = options(tmp, "jitter", max_samples=3)
    code = sampler.sample_run(opts, clock=LateClock(WALL0, MONO0))
    require(code == 0, "jitter run")
    samples = load_samples(run_dir(tmp, "jitter"))
    require(samples[0]["jitter_ns"] is None, "first jitter stays null")
    require(samples[1]["jitter_ns"] == 2_000_000, "late interval is recorded")


def check_max_duration(tmp: Path) -> None:
    opts = options(tmp, "bounded", max_samples=10, max_duration_s=1.5, cadence_ms=1000)
    code = sampler.sample_run(opts, clock=clock())
    require(code == 0, "bounded run")
    samples = load_samples(run_dir(tmp, "bounded"))
    require(len(samples) == 2, f"duration bound should stop at 2 samples, got {len(samples)}")


def check_vram_cache_resets(tmp: Path) -> None:
    opts = options(tmp, "cache", fake=None, max_samples=3)
    code = sampler.sample_run(opts, backend=CachingFree(), clock=clock())
    require(code == 0, "cache run")
    samples = load_samples(run_dir(tmp, "cache"))
    frees = [sample["vram_free"]["value"] for sample in samples]
    require(len(set(frees)) == len(frees), f"each sample must re-read VRAM, got {frees}")
    require(frees == sorted(frees, reverse=True), f"VRAM readings must change per sample, got {frees}")


def check_bad_run_id(tmp: Path) -> None:
    try:
        sampler.sample_run(options(tmp, "../escape"), clock=clock())
    except sampler.SamplerError:
        return
    raise CheckError("path-like run id must be rejected")


def check_second_start_rejected(tmp: Path) -> None:
    opts = options(tmp, "once", max_samples=1)
    require(sampler.sample_run(opts, clock=clock()) == 0, "first start")
    try:
        sampler.sample_run(options(tmp, "once", max_samples=1), clock=clock())
    except sampler.SamplerError:
        return
    raise CheckError("a second start for the same run id must fail")


_SIGNAL_CHILD = (
    "import os,signal,sys,time\nfrom pathlib import Path\n"
    "path, name = Path(sys.argv[1]), sys.argv[2]\n"
    "signum = signal.SIGINT if name == 'INT' else signal.SIGTERM\nseen = 0\n"
    "def on_signal(_signum, _frame):\n global seen\n seen += 1\n"
    " path.write_text(f'got {seen} {os.getpgrp()}')\n"
    " signal.signal(signum, signal.SIG_DFL)\n os.kill(os.getpid(), signum)\n"
    "signal.signal(signum, on_signal)\n"
    "path.write_text(f'ready {os.getpid()} {os.getpgrp()}')\ntime.sleep(30)\n"
)


def signal_when_ready(path: Path, signum: int) -> None:
    deadline = time.monotonic() + 5
    text = ""
    while time.monotonic() < deadline and not text.startswith("ready "):
        text = path.read_text(encoding="utf-8") if path.is_file() else ""
        time.sleep(0.02)
    if not text.startswith("ready "):
        raise CheckError(f"child did not become ready for signal {signum}")
    if signum == signal.SIGINT:
        os.killpg(os.getpgrp(), signum)
        return
    os.kill(os.getpid(), signum)


def run_cli(tmp: Path, run_id: str, *command: str) -> int:
    return sampler.main(
        [
            "f0_gpu_sampler.py", "run", "--run-id", run_id, "--out", str(tmp),
            "--fake", "full-support", "--hostname", "ShipOfTheseus",
            "--cadence-ms", "50", "--max-duration-s", "8", "--", *command,
        ]
    )


def run_signaled(tmp: Path, run_id: str, signum: int, expected: int) -> None:
    marker = tmp / f"{run_id}.txt"
    name = "INT" if signum == signal.SIGINT else "TERM"
    threading.Thread(target=signal_when_ready, args=(marker, signum), daemon=True).start()
    code = run_cli(tmp, run_id, sys.executable, "-c", _SIGNAL_CHILD, str(marker), name)
    require(code == expected, f"{name} exit {code}, expected {expected}")
    recorded = marker.read_text(encoding="utf-8")
    count, child_group = recorded.split()[1:]
    require(count == "1" and int(child_group) == os.getpgrp(), f"{name} delivery {recorded}")
    skew = run_dir(tmp, run_id) / "clock-skew.json"
    require(skew.is_file(), f"{name} final record missing")


def check_tty_stdin_read(tmp: Path) -> None:
    marker = tmp / "tty-read.txt"
    script = "import sys\nfrom pathlib import Path\nPath(sys.argv[1]).write_text(sys.stdin.readline())\n"
    script_path = str(Path(sampler.__file__).resolve())
    code = run_with_controlling_tty(
        ["python3", script_path, "run", "--run-id", "ttyread", "--out", str(tmp),
         "--fake", "full-support", "--hostname", "ShipOfTheseus", "--cadence-ms", "50",
         "--max-duration-s", "4", "--", sys.executable, "-c", script, str(marker)],
        b"hello\n",
    )
    require(code == 0 and marker.read_text(encoding="utf-8") == "hello\n", f"tty read {code}")


def check_forward_sigint(tmp: Path) -> None:
    if os.getpgrp() != os.getpid():
        os.setpgrp()
    run_signaled(tmp, "sigint", signal.SIGINT, 128 + signal.SIGINT)


def check_forward_sigterm(tmp: Path) -> None:
    run_signaled(tmp, "sigterm", signal.SIGTERM, 128 + signal.SIGTERM)


def check_sigpipe_head(tmp: Path) -> None:
    code, err = run_writing_to_head(lambda: run_cli(tmp, "sigpipe", "yes"))
    quiet = code == 128 + signal.SIGPIPE and b"Traceback" not in err and b"Broken pipe" not in err
    require(quiet and (run_dir(tmp, "sigpipe") / "clock-skew.json").is_file(), f"sigpipe {code} {err!r}")


def check_ctrl_z_fg(tmp: Path) -> None:
    # The AND list has to be one job. An interactive shell treats a stop as status
    # 148 and would skip a top-level `&& echo`.
    # The trainer touches a ready file before sleeping, so Ctrl+Z lands while
    # `run` is in wait_child rather than during Python startup.
    tool = str(Path(sampler.__file__).resolve())
    ready = tmp / "sigtstp.ready"
    command = (
        "bash -c '"
        f"python3 {tool} run --run-id sigtstp --out {tmp} --fake full-support "
        "--hostname ShipOfTheseus --cadence-ms 50 --max-duration-s 12 -- "
        f'sh -c "touch {ready}; exec sleep 2" && echo SURVIVED\'\n'
    )
    output = bash_stop_and_fg(command.encode(), ready)
    require(ready.exists(), "trainer never reached its ready point")
    lines = [line.strip() for line in output.replace(b"\r\n", b"\n").replace(b"\r", b"\n").split(b"\n")]
    require(b"Stopped" in output and b"SURVIVED" in lines, f"ctrl-z fg {output!r}")
    require((run_dir(tmp, "sigtstp") / "clock-skew.json").is_file(), "ctrl-z final record")


def check_run_command_exit(tmp: Path) -> None:
    code = run_cli(tmp, "boom", sys.executable, "-c", "import sys; sys.exit(9)")
    require(code == 9, f"command exit code must propagate, got {code}")
    directory = run_dir(tmp, "boom")
    samples = load_samples(directory)
    require(samples, "a failed command still keeps the samples")
    skew = json.loads((directory / "clock-skew.json").read_text(encoding="utf-8"))
    f0_clock.check_clock_skew_record(skew)
    names = [thread.name for thread in threading.enumerate()]
    require("bkl-gpu-sampler" not in names, "sampler thread exited with the command")
    session = json.loads((directory / "session.json").read_text(encoding="utf-8"))
    require(session["status"] == "stopped", "session is stopped after the command")


def wait_for_sample(path: Path) -> None:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if path.is_file() and path.stat().st_size > 0:
            return
        time.sleep(0.02)
    raise CheckError("sampler process produced no sample")


def check_start_stop_process(tmp: Path) -> None:
    base = [
        "f0_gpu_sampler.py", "--run-id", "proc", "--out", str(tmp), "--fake", "full-support",
        "--hostname", "ShipOfTheseus", "--cadence-ms", "30", "--max-duration-s", "4", "--max-samples", "40",
    ]
    code = sampler.main([base[0], "start", *base[1:]])
    directory = run_dir(tmp, "proc")
    try:
        require(code == 0, "start exit")
        session = json.loads((directory / "session.json").read_text(encoding="utf-8"))
        pid = session["pid"]
        require(isinstance(pid, int) and sampler.pid_alive(pid), "sampler process is running")
        wait_for_sample(directory / "samples.jsonl")
        stopped = sampler.main([base[0], "stop", "--run-id", "proc", "--out", str(tmp)])
        require(stopped == 0 and not sampler.pid_alive(pid), "stop exits the sampler process")
    finally:
        sampler.main([base[0], "stop", "--run-id", "proc", "--out", str(tmp)])
    samples = load_samples(directory)
    require(samples, "process wrote samples")
    for sample in samples:
        f0corr.check_sample(sample)
        assert_null_when_missing(sample)
    skew = json.loads((directory / "clock-skew.json").read_text(encoding="utf-8"))
    f0_clock.check_clock_skew_record(skew)
    again = sampler.main([base[0], "stop", "--run-id", "proc", "--out", str(tmp)])
    require(again == 0, "second stop is safe")
    session = json.loads((directory / "session.json").read_text(encoding="utf-8"))
    require(session["status"] == "stopped" and session["pid"] is None, "session stopped")


def check_join_phase_from_marker(tmp: Path) -> None:
    opts = options(tmp, "full-join", fake="full-support")
    require(sampler.sample_run(opts, clock=clock()) == 0, "join run")
    _snapshot, samples = assert_bundle(run_dir(tmp, "full-join"))
    markers_path = run_dir(tmp, "full-join") / "markers.jsonl"
    write_markers(markers_path, samples)
    markers = f0corr.load_jsonl(markers_path)
    pairs = f0corr.join_samples(markers, samples)
    require(len(pairs) == len(samples), "every sample joins a marker")
    require(all(marker["phase"] == "train" for marker, _sample in pairs), "phase comes from the marker")


def check_stop_during_read_drops_sample(tmp: Path) -> None:
    event = threading.Event()
    backend = StopsMidRead(event)
    opts = options(tmp, "stop-mid-read", max_samples=5)
    directory = run_dir(tmp, "stop-mid-read")
    sampler.ensure_new_run(directory)
    armed = sampler.arm(opts, backend, clock(), directory)
    require(armed.refused is None, f"arm refused: {armed.refused}")
    state = sampler.loop_state_for(armed, opts, backend, clock(), event)
    backend.armed = True
    count = sampler.sample_loop(state)
    require(count == 0, f"sample read across a stop must be dropped, loop counted {count}")
    require(load_samples(directory) == [], "no sample may be appended after stop was requested")


def check_atomic_session_write(tmp: Path) -> None:
    target = tmp / "atomic" / "session.json"
    sampler.write_json(target, {"status": "running", "pid": None})
    # A reader that opened the file before the rewrite must still see the whole
    # old object. An in-place truncate+write would hand it an empty or new body.
    with target.open(encoding="utf-8") as early_reader:
        sampler.write_json(target, {"status": "running", "pid": 4242})
        require(json.loads(early_reader.read())["pid"] is None, "rewrite must replace, not truncate in place")
    require(sampler.read_json(target)["pid"] == 4242, "rewrite must land")
    leftovers = [path.name for path in target.parent.iterdir() if path.name != "session.json"]
    require(not leftovers, f"atomic write left temp files: {leftovers}")


CASES = (
    ("stat_state_and_missing_headroom", check_stat_state_and_missing_headroom),
    ("floor_constant", check_floor_constant),
    ("full_support", check_full_support),
    ("partial_support", check_partial_support),
    ("missing_reading", check_missing_reading),
    ("permission_denied", check_permission_denied),
    ("device_lost", check_device_lost),
    ("valid_zero", check_valid_zero),
    ("no_device", check_no_device),
    ("low_headroom", check_low_headroom),
    ("overhead_budget", check_overhead_budget),
    ("jitter_recorded", check_jitter_recorded),
    ("max_duration", check_max_duration),
    ("vram_cache_resets", check_vram_cache_resets),
    ("bad_run_id", check_bad_run_id),
    ("second_start_rejected", check_second_start_rejected),
    ("tty_stdin_read", check_tty_stdin_read),
    ("forward_sigint", check_forward_sigint),
    ("forward_sigterm", check_forward_sigterm),
    ("sigpipe_head", check_sigpipe_head),
    ("ctrl_z_fg", check_ctrl_z_fg),
    ("run_command_exit", check_run_command_exit),
    ("start_stop_process", check_start_stop_process),
    ("join_phase_from_marker", check_join_phase_from_marker),
    ("stop_during_read_drops_sample", check_stop_during_read_drops_sample),
    ("atomic_session_write", check_atomic_session_write),
)


def main(argv: list[str]) -> int:
    del argv
    failed = 0
    with tempfile.TemporaryDirectory(prefix="bkl-f0-sampler-") as raw:
        tmp = Path(raw)
        for name, fn in CASES:
            try:
                if name in {"floor_constant", "stat_state_and_missing_headroom"}:
                    fn()
                else:
                    fn(tmp)
            except (
                CheckError,
                SchemaError,
                f0_clock.ClockError,
                sampler.SamplerError,
                OSError,
                KeyError,
                TypeError,
                json.JSONDecodeError,
                AssertionError,
            ) as error:
                failed += 1
                print(f"FAIL {name}: {error}", file=sys.stderr)
            else:
                print(f"ok {name}")
    if failed:
        print(f"{failed} failed", file=sys.stderr)
        return 1
    print(f"{len(CASES)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
