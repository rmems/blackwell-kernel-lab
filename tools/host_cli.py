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


def execute(
    argv: list[str],
    *,
    cwd: Path | None = None,
    timeout: float | None = None,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run a list argv. ``capture`` keeps stdio for callers that check it."""
    return subprocess.run(  # nosec B603 — list argv, never a shell string
        argv,
        check=False,
        cwd=cwd,
        timeout=timeout,
        capture_output=capture,
        text=True,
    )


def spawn(
    argv: list[str],
    *,
    cwd: Path,
) -> subprocess.Popen[bytes]:
    """Start a list argv in its own session so the parent can exit."""
    return subprocess.Popen(  # nosec B603 — list argv, new session, no shell
        argv,
        cwd=cwd,
        start_new_session=True,
    )


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


