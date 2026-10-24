"""Test-only harnesses for ``check_f0_gpu_sampler.py``.

Pty, pipe, and job-shell drivers plus small NVML fake wrappers. The sampler
itself never imports this module. CPU only; no CUDA context.
"""

from __future__ import annotations

import fcntl
import os
import pty
import select
import signal
import termios
import threading
import time
from pathlib import Path
from typing import Any

from f0_sampler_support import exec_python, exit_code_from_status, signal_pid
from nvml_fakes import FakeNvmlBackend


class CachingFree:
    """Returns a new free-MiB reading only after begin_reading."""

    def __init__(self) -> None:
        self.inner = FakeNvmlBackend("full-support")
        self.cached: int | None = None
        self.generation = 14000

    def begin_reading(self) -> None: self.cached = None

    def gpu_count(self) -> int: return self.inner.gpu_count()

    def identity(self, index: int) -> Any: return self.inner.identity(index)

    def versions(self) -> Any: return self.inner.versions()

    def read_throttle(self, index: int) -> list[str]: return self.inner.read_throttle(index)

    def read_numeric(self, index: int, metric: str) -> int | float:
        if metric != "vram_free":
            return self.inner.read_numeric(index, metric)
        if self.cached is None:
            self.generation -= 10
            self.cached = self.generation
        return self.cached


class StopsMidRead:
    """Sets the stop event during the first physical read, as a stop racing a slow NVML call would."""

    def __init__(self, event: threading.Event) -> None:
        self.inner = FakeNvmlBackend("full-support")
        self.event = event
        self.armed = False

    def gpu_count(self) -> int: return self.inner.gpu_count()

    def identity(self, index: int) -> Any: return self.inner.identity(index)

    def versions(self) -> Any: return self.inner.versions()

    def read_throttle(self, index: int) -> list[str]: return self.inner.read_throttle(index)

    def read_numeric(self, index: int, metric: str) -> int | float:
        if self.armed:
            self.event.set()
        return self.inner.read_numeric(index, metric)


def enter_controlling_tty(slave: int, master: int, command: list[str]) -> None:
    os.setsid(); os.close(master)
    try:
        fcntl.ioctl(slave, termios.TIOCSCTTY, 0)
    except OSError:
        os._exit(127)
    os.dup2(slave, 0); os.dup2(slave, 1); os.dup2(slave, 2)
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


def _exec_head(read_fd: int, write_fd: int) -> None:
    os.dup2(read_fd, 0)
    os.close(read_fd)
    os.close(write_fd)
    null_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null_fd, 1)
    os.dup2(null_fd, 2)
    os.close(null_fd)
    try:
        os.execvp("head", ["head", "-n", "1"])  # nosec B606 — literal head, not a shell
    except OSError:
        os._exit(127)


def _restore_fd(target: int, saved: int) -> None:
    os.dup2(saved, target)
    os.close(saved)


def _reap_pid(pid: int) -> None:
    signal_pid(pid, signal.SIGKILL)
    try:
        os.waitpid(pid, 0)
    except ChildProcessError:
        return


def run_writing_to_head(invoke: Any) -> tuple[int, bytes]:
    """Run invoke with stdout piped to ``head -n 1`` and capture stderr."""
    read_fd, write_fd = os.pipe()
    err_read, err_write = os.pipe()
    head = os.fork()
    if head == 0:
        os.close(err_read); os.close(err_write); _exec_head(read_fd, write_fd)
    os.close(read_fd)
    saved_out, saved_err = os.dup(1), os.dup(2)
    os.dup2(write_fd, 1); os.dup2(err_write, 2)
    os.close(write_fd); os.close(err_write)
    try:
        code = int(invoke())
    finally:
        _restore_fd(1, saved_out); _restore_fd(2, saved_err); _reap_pid(head)
        err = os.read(err_read, 65536); os.close(err_read)
    return code, err


def _exec_job_shell(slave: int, master: int) -> None:
    try:
        os.setsid()
        os.close(master)
        fcntl.ioctl(slave, termios.TIOCSCTTY, 0)
        os.dup2(slave, 0)
        os.dup2(slave, 1)
        os.dup2(slave, 2)
        if slave > 2:
            os.close(slave)
        os.environ["PS1"] = "$ "
        os.execvp("bash", ["bash", "--noprofile", "--norc", "-im"])  # nosec B606 — literal bash
    except OSError:
        os._exit(127)


def _drain_pty(master: int, seconds: float, chunks: list[bytes], marker: bytes | None) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        ready, _, _ = select.select([master], [], [], 0.05)
        if ready: chunks.append(os.read(master, 4096))
        if marker is not None and marker in b"".join(chunks): return


def _drain_until_file(master: int, seconds: float, chunks: list[bytes], ready: Path) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline and not ready.exists():
        _drain_pty(master, 0.05, chunks, None)


def bash_stop_and_fg(line: bytes, ready: Path | None = None) -> bytes:
    """Run line in an interactive job shell, Ctrl+Z it, then ``fg``.

    With ``ready``, Ctrl+Z waits until that file exists (the trainer has
    started) instead of a fixed delay, so the stop lands in ``wait_child``.
    """
    master, slave = pty.openpty()
    pid = os.fork()
    if pid == 0:
        _exec_job_shell(slave, master)
    os.close(slave)
    chunks: list[bytes] = []
    os.write(master, line)
    if ready is None:
        _drain_pty(master, 0.7, chunks, None)
    else:
        _drain_until_file(master, 10.0, chunks, ready)
        _drain_pty(master, 0.1, chunks, None)
    os.write(master, b"\x1a")
    _drain_pty(master, 2.0, chunks, b"Stopped")
    os.write(master, b"fg\n")
    _drain_pty(master, 5.0, chunks, b"\nSURVIVED")
    os.kill(pid, signal.SIGKILL)
    os.waitpid(pid, 0)
    os.close(master)
    return b"".join(chunks)
