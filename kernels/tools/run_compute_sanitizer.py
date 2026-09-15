#!/usr/bin/env python3
"""Run NVIDIA Compute Sanitizer on first-party CUDA binaries.

Used by .github/workflows/ci-gpu-sanitizer.yml and the local recipe in
recipes/compute-sanitizer.md. CPU tests exercise this file with fake tools;
they do not need a GPU.

Usage:
    kernels/tools/run_compute_sanitizer.py --suite \\
        --bin-dir build/kernels/src --out-dir results/sanitizer
    kernels/tools/run_compute_sanitizer.py --tool memcheck --out-dir DIR -- \\
        ./build/kernels/src/bkl_device_hello
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = "bkl.compute_sanitizer.v1"
SUITE_SCHEMA = "bkl.compute_sanitizer.suite.v1"
TOOLS = ("memcheck", "initcheck", "racecheck", "synccheck")
PRINT_LIMIT = 50
MAX_LOG_BYTES = 1 << 20
PREVIEW_BYTES = 64 * 1024
ERROR_EXITCODE = 1
PER_JOB_TIMEOUT_S = 900
ERROR_SUMMARY_RE = re.compile(r"ERROR SUMMARY:\s+(\d+)\s+errors?\b")
CUDA_BIN = Path("/usr/local/cuda/bin")

# Keep this list in lockstep with the fake sanitizer in
# tools/check_compute_sanitizer_runner.py.
SANITIZER_FLAG_SPECS = (
    ("--tool", "{tool}"),
    ("--error-exitcode", str(ERROR_EXITCODE)),
    ("--print-limit", str(PRINT_LIMIT)),
    ("--check-exit-code", "yes"),
)


@dataclass(frozen=True)
class Job:
    tool: str
    binary: Path
    args: tuple[str, ...]


class RunnerError(Exception):
    """A sanitizer invocation or host probe failed."""


def require(condition: object, message: str) -> None:
    if not condition:
        raise RunnerError(message)


def find_cuda_tool(name: str) -> str | None:
    env_key = name.upper().replace("-", "_")
    override = os.environ.get(env_key)
    if override:
        return override
    found = shutil.which(name)
    if found:
        return found
    candidate = CUDA_BIN / name
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
    return None


def capture_text(cmd: list[str]) -> str:
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except FileNotFoundError:
        return ""
    return (proc.stdout or "") + (proc.stderr or "")


def bound_text(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "\n[truncated]\n"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_error_summary(text: str) -> int | None:
    matches = ERROR_SUMMARY_RE.findall(text)
    if not matches:
        return None
    return int(matches[-1])


def parse_gpu_csv(raw: str) -> dict[str, str | None]:
    line = raw.strip().splitlines()[0] if raw.strip() else ""
    parts = [part.strip() for part in line.split(",")]
    name = parts[0] if len(parts) > 0 and parts[0] else None
    driver = parts[1] if len(parts) > 1 and parts[1] else None
    cap = parts[2] if len(parts) > 2 and parts[2] else None
    return {"name": name, "driver_version": driver, "compute_cap": cap}


def host_snapshot(sanitizer: str) -> dict[str, Any]:
    nvcc = find_cuda_tool("nvcc")
    nvidia_smi = find_cuda_tool("nvidia-smi")
    gpu = {"name": None, "driver_version": None, "compute_cap": None}
    if nvidia_smi:
        gpu = parse_gpu_csv(
            capture_text(
                [
                    nvidia_smi,
                    "--query-gpu=name,driver_version,compute_cap",
                    "--format=csv,noheader",
                ]
            )
        )
    nvcc_version = bound_text(capture_text([nvcc, "--version"]), 512) if nvcc else ""
    sanitizer_version = bound_text(capture_text([sanitizer, "--version"]), 512)
    return {
        "nvcc_version": nvcc_version or None,
        "sanitizer_version": sanitizer_version or None,
        "gpu_name": gpu["name"],
        "driver_version": gpu["driver_version"],
        "compute_cap": gpu["compute_cap"],
    }


def sanitizer_command(sanitizer: str, tool: str, binary: Path, args: list[str]) -> list[str]:
    cmd = [sanitizer]
    for flag, value in SANITIZER_FLAG_SPECS:
        cmd.extend([flag, value.format(tool=tool)])
    cmd.append(str(binary))
    cmd.extend(args)
    return cmd


def maybe_truncate(path: Path, max_bytes: int) -> bool:
    size = path.stat().st_size
    if size <= max_bytes:
        return False
    keep = max(0, max_bytes - 32)
    data = path.read_bytes()[:keep]
    path.write_bytes(data + b"\n[truncated]\n")
    return True


def preview_log(path: Path) -> None:
    data = path.read_bytes()[:PREVIEW_BYTES]
    sys.stdout.buffer.write(data)
    if path.stat().st_size > PREVIEW_BYTES:
        sys.stdout.write("\n[job-log preview truncated]\n")
    sys.stdout.flush()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def run_one(
    job: Job,
    out_dir: Path,
    sanitizer: str,
    host: dict[str, Any],
    max_log_bytes: int,
) -> dict[str, Any]:
    require(job.tool in TOOLS, f"unsupported sanitizer tool: {job.tool}")
    require(job.binary.is_file(), f"missing binary: {job.binary}")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{job.tool}-{job.binary.name}"
    log_path = out_dir / f"{stem}.log"
    meta_path = out_dir / f"{stem}.json"
    command = sanitizer_command(sanitizer, job.tool, job.binary, list(job.args))
    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            proc = subprocess.run(
                command,
                check=False,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                timeout=PER_JOB_TIMEOUT_S,
            )
    except subprocess.TimeoutExpired as error:
        raise RunnerError(
            f"{job.tool} {job.binary.name}: timed out after {error.timeout}s"
        ) from error
    log_text = log_path.read_text(errors="replace")
    error_summary = parse_error_summary(log_text)
    truncated = maybe_truncate(log_path, max_log_bytes)
    payload = {
        "schema": SCHEMA,
        "tool": job.tool,
        "sanitizer_version": host["sanitizer_version"],
        "cuda": {"nvcc_version": host["nvcc_version"]},
        "driver_version": host["driver_version"],
        "gpu": {"name": host["gpu_name"], "compute_cap": host["compute_cap"]},
        "binary": {"path": str(job.binary), "sha256": sha256_file(job.binary)},
        "command": command,
        "exit_code": proc.returncode,
        "error_summary": error_summary,
        "log_path": str(log_path),
        "log_bytes": log_path.stat().st_size,
        "log_truncated": truncated,
    }
    write_json(meta_path, payload)
    print(f"wrote {meta_path}")
    preview_log(log_path)
    if error_summary is None:
        raise RunnerError(f"{log_path}: missing ERROR SUMMARY")
    if error_summary != 0 or proc.returncode != 0:
        raise RunnerError(
            f"{job.tool} {job.binary.name}: exit_code={proc.returncode} "
            f"error_summary={error_summary}"
        )
    return payload


def suite_jobs(bin_dir: Path, out_dir: Path) -> list[Job]:
    hello = bin_dir / "bkl_device_hello"
    graph = bin_dir / "bkl_graph_launch_bench"
    green = bin_dir / "bkl_green_ctx_bench"
    return [
        Job("memcheck", hello, ()),
        Job("initcheck", hello, ()),
        Job(
            "memcheck",
            graph,
            ("--smoke", "--out", str(out_dir / "graph-launch-memcheck.json")),
        ),
        Job(
            "initcheck",
            graph,
            ("--smoke", "--out", str(out_dir / "graph-launch-initcheck.json")),
        ),
        Job(
            "memcheck",
            green,
            ("--smoke", "--out", str(out_dir / "green-ctx-memcheck.json")),
        ),
    ]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", action="store_true")
    parser.add_argument("--bin-dir", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tool", choices=TOOLS, default="memcheck")
    parser.add_argument("--max-log-bytes", type=int, default=MAX_LOG_BYTES)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser.parse_args(argv)


def command_job(tool: str, command: list[str]) -> Job:
    if command and command[0] == "--":
        command = command[1:]
    require(command, "pass a binary after --")
    return Job(tool, Path(command[0]), tuple(command[1:]))


def run_jobs(jobs: list[Job], out_dir: Path, max_log_bytes: int) -> int:
    sanitizer = find_cuda_tool("compute-sanitizer")
    require(sanitizer, "compute-sanitizer not found on PATH or /usr/local/cuda/bin")
    host = host_snapshot(sanitizer)
    require(host["sanitizer_version"], "compute-sanitizer --version produced no output")
    results: list[dict[str, Any]] = []
    failed: list[str] = []
    for job in jobs:
        try:
            results.append(run_one(job, out_dir, sanitizer, host, max_log_bytes))
        except RunnerError as error:
            failed.append(str(error))
            print(error, file=sys.stderr)
    summary = {"schema": SUITE_SCHEMA, "runs": results, "failures": failed}
    write_json(out_dir / "summary.json", summary)
    if failed:
        print(f"{len(failed)} sanitizer job(s) failed", file=sys.stderr)
        return 1
    print(f"{len(results)} sanitizer job(s) clean")
    return 0


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        if args.suite:
            require(args.bin_dir is not None, "--suite requires --bin-dir")
            jobs = suite_jobs(args.bin_dir, args.out_dir)
        else:
            jobs = [command_job(args.tool, args.command)]
        return run_jobs(jobs, args.out_dir, args.max_log_bytes)
    except RunnerError as error:
        print(error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
