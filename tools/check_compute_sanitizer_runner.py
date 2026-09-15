#!/usr/bin/env python3
"""CPU tests for kernels/tools/run_compute_sanitizer.py.

Creates fake compute-sanitizer / nvcc / nvidia-smi binaries and a dummy
target so GitHub-hosted CI can prove: clean logs pass, findings fail,
ERROR SUMMARY is required, logs are truncated, and metadata is recorded.

Usage:
    python3 tools/check_compute_sanitizer_runner.py
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "kernels" / "tools" / "run_compute_sanitizer.py"
FIXTURES = ROOT / "fixtures" / "compute-sanitizer"

FAKE_SANITIZER = r"""#!/usr/bin/env python3
import os
import subprocess
import sys

if "--version" in sys.argv:
    print("NVIDIA Compute Sanitizer, Version 2025.3.0")
    raise SystemExit(0)

args = sys.argv[1:]
skip_value = {
    "--tool",
    "--error-exitcode",
    "--print-limit",
    "--check-exit-code",
}
i = 0
while i < len(args):
    arg = args[i]
    if arg in skip_value:
        i += 2
        continue
    if arg == "--":
        args = args[i + 1 :]
        break
    if arg.startswith("-"):
        raise SystemExit(f"unexpected sanitizer flag: {arg}")
    args = args[i:]
    break
else:
    args = []

if not args:
    raise SystemExit("fake compute-sanitizer: missing application")

pad = os.environ.get("FAKE_SANITIZER_PAD", "")
if pad:
    sys.stdout.write(pad)
    sys.stdout.flush()

proc = subprocess.run(args, check=False)
errors = os.environ.get("FAKE_SANITIZER_ERRORS", "0")
noun = "error" if errors == "1" else "errors"
print(f"========= COMPUTE-SANITIZER")
if os.environ.get("FAKE_SANITIZER_OMIT_SUMMARY") == "1":
    raise SystemExit(proc.returncode)
print(f"========= ERROR SUMMARY: {errors} {noun}")
if int(errors) > 0:
    raise SystemExit(int(os.environ.get("FAKE_SANITIZER_EXIT", "1")))
raise SystemExit(proc.returncode)
"""

FAKE_NVCC = """#!/bin/sh
echo 'nvcc: NVIDIA (R) Cuda compiler driver'
echo 'Cuda compilation tools, release 13.3, V13.3.73'
"""

FAKE_NVIDIA_SMI = """#!/bin/sh
echo 'NVIDIA GeForce RTX 5080, 610.43.03, 12.0'
"""

FAKE_APP = """#!/bin/sh
echo 'bkl device_hello sm_120 ok'
"""


class TestError(Exception):
    """A runner self-test failed."""


def require(condition: object, message: str) -> None:
    if not condition:
        raise TestError(message)


def write_exec(path: Path, contents: str) -> None:
    path.write_text(contents)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def make_fakes(tmp: Path) -> Path:
    bindir = tmp / "bin"
    bindir.mkdir()
    write_exec(bindir / "compute-sanitizer", FAKE_SANITIZER)
    write_exec(bindir / "nvcc", FAKE_NVCC)
    write_exec(bindir / "nvidia-smi", FAKE_NVIDIA_SMI)
    write_exec(bindir / "bkl_device_hello", FAKE_APP)
    write_exec(bindir / "bkl_graph_launch_bench", FAKE_APP)
    write_exec(bindir / "bkl_green_ctx_bench", FAKE_APP)
    return bindir


def runner_env(bindir: Path, extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = str(bindir) + os.pathsep + env.get("PATH", "")
    env.pop("COMPUTE_SANITIZER", None)
    env.pop("NVCC", None)
    env.pop("NVIDIA_SMI", None)
    if extra:
        env.update(extra)
    return env


def run_runner(bindir: Path, args: list[str], extra: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(RUNNER), *args],
        check=False,
        capture_output=True,
        text=True,
        env=runner_env(bindir, extra),
    )


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def check_clean_one(tmp: Path, bindir: Path) -> None:
    out = tmp / "clean"
    proc = run_runner(
        bindir,
        ["--out-dir", str(out), "--", str(bindir / "bkl_device_hello")],
    )
    require(proc.returncode == 0, f"clean run failed: {proc.stderr}")
    meta = load_json(out / "memcheck-bkl_device_hello.json")
    require(meta["schema"] == "bkl.compute_sanitizer.v1", "missing metadata schema")
    require(meta["error_summary"] == 0, "clean run must record 0 errors")
    require(meta["exit_code"] == 0, "clean run must record exit 0")
    require(meta["tool"] == "memcheck", "default tool is memcheck")
    require(meta["binary"]["sha256"], "binary digest missing")
    require(meta["sanitizer_version"] and "Compute Sanitizer" in meta["sanitizer_version"],
            "sanitizer version missing")
    require(meta["cuda"]["nvcc_version"] and "13.3" in meta["cuda"]["nvcc_version"],
            "nvcc version missing")
    require(meta["gpu"]["name"] and "RTX 5080" in meta["gpu"]["name"], "GPU name missing")
    require(meta["driver_version"] == "610.43.03", "driver version missing")
    require("--error-exitcode" in meta["command"], "command must include --error-exitcode")
    require(not meta["log_truncated"], "clean fixture log should not truncate")


def check_findings_fail(tmp: Path, bindir: Path) -> None:
    out = tmp / "finding"
    proc = run_runner(
        bindir,
        ["--out-dir", str(out), "--", str(bindir / "bkl_device_hello")],
        extra={"FAKE_SANITIZER_ERRORS": "1"},
    )
    require(proc.returncode != 0, "findings must fail the runner")
    require("error_summary=1" in proc.stderr, f"stderr should mention findings: {proc.stderr}")
    meta = load_json(out / "memcheck-bkl_device_hello.json")
    require(meta["error_summary"] == 1, "finding metadata must record 1 error")


def check_missing_summary_fails(tmp: Path, bindir: Path) -> None:
    out = tmp / "nosummary"
    proc = run_runner(
        bindir,
        ["--out-dir", str(out), "--", str(bindir / "bkl_device_hello")],
        extra={"FAKE_SANITIZER_OMIT_SUMMARY": "1"},
    )
    require(proc.returncode != 0, "missing ERROR SUMMARY must fail")
    require("missing ERROR SUMMARY" in proc.stderr, proc.stderr)


def check_log_truncated(tmp: Path, bindir: Path) -> None:
    out = tmp / "trunc"
    pad = "x" * 400
    proc = run_runner(
        bindir,
        [
            "--out-dir",
            str(out),
            "--max-log-bytes",
            "128",
            "--",
            str(bindir / "bkl_device_hello"),
        ],
        extra={"FAKE_SANITIZER_PAD": pad},
    )
    require(proc.returncode == 0, f"truncated clean run failed: {proc.stderr}")
    meta = load_json(out / "memcheck-bkl_device_hello.json")
    require(meta["log_truncated"] is True, "log_truncated should be true")
    require(meta["log_bytes"] <= 160, f"log still too large: {meta['log_bytes']}")
    log = Path(meta["log_path"]).read_text()
    require("[truncated]" in log, "truncated marker missing")


def check_suite(tmp: Path, bindir: Path) -> None:
    out = tmp / "suite"
    proc = run_runner(
        bindir,
        ["--suite", "--bin-dir", str(bindir), "--out-dir", str(out)],
    )
    require(proc.returncode == 0, f"suite failed: {proc.stderr}\n{proc.stdout}")
    summary = load_json(out / "summary.json")
    require(summary["schema"] == "bkl.compute_sanitizer.suite.v1", "suite schema")
    require(len(summary["runs"]) == 5, f"expected 5 suite jobs, got {len(summary['runs'])}")
    tools = {(run["tool"], Path(run["binary"]["path"]).name) for run in summary["runs"]}
    require(("memcheck", "bkl_device_hello") in tools, "suite missing hello memcheck")
    require(("initcheck", "bkl_device_hello") in tools, "suite missing hello initcheck")
    require(("memcheck", "bkl_graph_launch_bench") in tools, "suite missing graph memcheck")
    require(("initcheck", "bkl_graph_launch_bench") in tools, "suite missing graph initcheck")
    require(("memcheck", "bkl_green_ctx_bench") in tools, "suite missing green memcheck")


def check_fixtures_exist() -> None:
    for name in ("memcheck-clean.log", "memcheck-finding.log"):
        path = FIXTURES / name
        require(path.is_file(), f"missing fixture {path}")
    clean = (FIXTURES / "memcheck-clean.log").read_text()
    finding = (FIXTURES / "memcheck-finding.log").read_text()
    require("ERROR SUMMARY: 0 errors" in clean, "clean fixture contract")
    require("ERROR SUMMARY: 1 error" in finding, "finding fixture contract")
    require("bkl_oob_example" in finding, "finding fixture should describe the docs-only OOB")


def main() -> int:
    try:
        check_fixtures_exist()
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            bindir = make_fakes(tmp)
            check_clean_one(tmp, bindir)
            check_findings_fail(tmp, bindir)
            check_missing_summary_fails(tmp, bindir)
            check_log_truncated(tmp, bindir)
            check_suite(tmp, bindir)
    except TestError as error:
        print(error, file=sys.stderr)
        return 1
    print("compute sanitizer runner: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
