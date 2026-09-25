"""Resolve host executables and run subprocesses with fixed argv lists."""

from __future__ import annotations

from collections.abc import Mapping
import shutil
import subprocess  # nosec B404 — subprocess is confined to this audited helper module
from pathlib import Path
from typing import Any

DEFAULT_TIMEOUT_SECONDS = 120
CalledProcessError = subprocess.CalledProcessError
STDERR_DEVNULL = subprocess.DEVNULL


def _require_on_path(name: str) -> None:
    if shutil.which(name) is None:
        raise FileNotFoundError(f"host executable not found: {name}")


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


def run_python_script(
    script: Path,
    *args: str,
    check: bool,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    **kwargs: Any,
) -> subprocess.CompletedProcess[str]:
    _require_on_path("python3")
    return subprocess.run(  # nosec B603 B607 — python3 on PATH verified; fixed argv
        ["python3", str(script.resolve()), *args],
        cwd=cwd,
        env=env,
        check=check,
        timeout=timeout,
        **kwargs,
    )
