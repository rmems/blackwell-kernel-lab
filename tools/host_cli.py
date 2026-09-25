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


def _require_on_path(name: str) -> None:
    if shutil.which(name) is None:
        raise FileNotFoundError(f"host executable not found: {name}")


def git_argv(*args: str) -> list[str]:
    _require_on_path("git")
    return ["git", *args]


def python_script_argv(script: Path, *args: str) -> list[str]:
    return [sys.executable, str(script.resolve()), *args]


def capture_git(
    *args: str,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    **kwargs: Any,
) -> bytes:
    _require_on_path("git")
    return subprocess.check_output(  # nosec B603 B607 — git on PATH verified; fixed argv
        ["git", *args],
        cwd=cwd,
        env=env,
        timeout=timeout,
        **kwargs,
    )


def run_git(
    *args: str,
    check: bool,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    **kwargs: Any,
) -> subprocess.CompletedProcess[str]:
    _require_on_path("git")
    return subprocess.run(  # nosec B603 B607 — git on PATH verified; fixed argv
        ["git", *args],
        cwd=cwd,
        env=env,
        check=check,
        timeout=timeout,
        **kwargs,
    )


def run_python_script(
    script: Path,
    *args: str,
    check: bool,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    **kwargs: Any,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # nosec B603 — sys.executable + resolved script path; no shell
        [sys.executable, str(script.resolve()), *args],
        cwd=cwd,
        env=env,
        check=check,
        timeout=timeout,
        **kwargs,
    )
