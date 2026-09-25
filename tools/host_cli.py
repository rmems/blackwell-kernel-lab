"""Resolve host executables and run subprocesses with fixed argv lists."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import shutil
import subprocess  # nosec B404 — subprocess is confined to this audited helper module
import sys
from pathlib import Path
from typing import Any

DEFAULT_TIMEOUT_SECONDS = 120
CalledProcessError = subprocess.CalledProcessError
STDERR_DEVNULL = subprocess.DEVNULL


def host_executable(name: str) -> str:
    resolved = shutil.which(name)
    if resolved is None:
        raise FileNotFoundError(f"host executable not found: {name}")
    return resolved


def python_script_argv(script: Path, *args: str) -> list[str]:
    return [sys.executable, str(script.resolve()), *args]


def git_argv(*args: str) -> list[str]:
    return [host_executable("git"), *args]


def run_checked(
    argv: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    **kwargs: Any,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # nosec B603 — argv uses resolved executables; no shell
        list(argv),
        cwd=cwd,
        env=env,
        check=True,
        timeout=timeout,
        **kwargs,
    )


def capture_checked(
    argv: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    **kwargs: Any,
) -> bytes:
    return subprocess.check_output(  # nosec B603 — argv uses resolved executables; no shell
        list(argv),
        cwd=cwd,
        env=env,
        timeout=timeout,
        **kwargs,
    )


def run_optional(
    argv: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    **kwargs: Any,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # nosec B603 — argv uses resolved executables; no shell
        list(argv),
        cwd=cwd,
        env=env,
        check=False,
        timeout=timeout,
        **kwargs,
    )
