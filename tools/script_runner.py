"""Run repository Python entrypoints in-process for deterministic unit tests."""

from __future__ import annotations

import importlib.util
import io
import os
import sys
from collections.abc import Mapping
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ScriptResult:
    returncode: int
    stdout: str
    stderr: str


def run_python_main(
    script: Path,
    *argv: str,
    env: Mapping[str, str],
    cwd: Path | None = None,
) -> ScriptResult:
    saved_cwd = Path.cwd()
    saved_env = os.environ.copy()
    saved_argv = sys.argv
    stdout = io.StringIO()
    stderr = io.StringIO()
    try:
        if cwd is not None:
            os.chdir(cwd)
        os.environ.clear()
        os.environ.update(env)
        spec = importlib.util.spec_from_file_location(f"script_{script.stem}", script)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load script module: {script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.argv = [script.name, *argv]
        with redirect_stdout(stdout), redirect_stderr(stderr):
            returncode = module.main()
    finally:
        sys.argv = saved_argv
        os.chdir(saved_cwd)
        os.environ.clear()
        os.environ.update(saved_env)
    return ScriptResult(returncode=returncode, stdout=stdout.getvalue(), stderr=stderr.getvalue())
