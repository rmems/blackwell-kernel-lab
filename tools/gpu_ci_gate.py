#!/usr/bin/env python3
"""Detect GPU-relevant changes and fail closed on required GPU job results."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys

GPU_POLICY = {
    "tools/gpu_ci_gate.py", "tools/check_gpu_ci_gate.py", "tools/check_gpu_ci_policy.py",
}
CPU_FILES = {
    ".gitignore", ".github/actionlint.yaml",
    ".github/workflows/ci-cpu.yml", ".github/workflows/ci-lint.yml",
    ".qlty/qlty.toml", ".qlty/.gitignore",
    "tools/check_f0_clock_skew.py", "tools/check_f0_correlation.py",
    "tools/check_f0_efficiency.py", "tools/check_nvml_capability.py",
    "tools/f0_clock.py", "tools/f0_efficiency_compare.py",
    "tools/f0_efficiency_numbers.py", "tools/f0_efficiency_report.py",
    "tools/f0_measurements.py", "tools/nvml_capability.py", "tools/nvml_fakes.py",
    "tools/nvml_live.py", "tools/nvml_schema.py", "tools/summarize_f0_efficiency.py",
}
DEVICE_SUFFIXES = {".cu", ".cuh", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".cmake"}
CPU_FIXTURES = {"f0-correlation", "f0-efficiency", "f0-nvml-capability"}


def needs_gpu(path: str) -> bool:
    """Only known documentation/CPU paths skip; unknown build/source paths run."""
    file = Path(path)
    if file.suffix == ".md" or path == "LICENSE":
        return False
    if path in GPU_POLICY or file.suffix in DEVICE_SUFFIXES or file.name == "CMakeLists.txt":
        return True
    if path in CPU_FILES:
        return False
    if (len(file.parts) >= 3 and file.parts[0] == "fixtures"
            and file.parts[1] in CPU_FIXTURES and file.suffix in {".json", ".jsonl"}):
        return False
    return True


def commit_sha(value: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", value):
        raise ValueError("missing or invalid event commit SHA")
    return value


def changed_paths(event: dict, event_name: str) -> list[str]:
    if event_name == "pull_request_target":
        pr = event["pull_request"]
        base, head = commit_sha(pr["base"]["sha"]), commit_sha(pr["head"]["sha"])
        revision_range = f"{base}...{head}"
    elif event_name == "push":
        base, head = commit_sha(event["before"]), commit_sha(event["after"])
        if set(base) == {"0"}:
            return ["new-branch-requires-gpu"]
        revision_range = f"{base}..{head}"
    else:
        raise ValueError("unsupported event; cannot classify GPU changes")
    # Disabling rename detection preserves both names of a CUDA → docs rename.
    output = subprocess.check_output(
        ["git", "diff", "--name-only", "--no-renames", "-z", revision_range, "--"]
    )
    return [name.decode("utf-8", errors="surrogateescape") for name in output.split(b"\0") if name]


def detect() -> None:
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    event_name = os.environ["GITHUB_EVENT_NAME"]
    repository = os.environ["GITHUB_REPOSITORY"]
    if not repository or event["repository"]["full_name"] != repository:
        raise ValueError("event repository identity mismatch")
    trusted = True
    if event_name == "pull_request_target":
        trusted = event["pull_request"]["head"]["repo"]["full_name"] == repository
    required = event_name == "workflow_dispatch" or any(
        needs_gpu(path) for path in changed_paths(event, event_name)
    )
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as stream:
        stream.write(f"required={str(required).lower()}\ntrusted={str(trusted).lower()}\n")
    print(f"GPU required: {required}; trusted repository: {trusted}")


def verify() -> None:
    if os.environ.get("DETECTION_RESULT") != "success":
        raise ValueError("change detection did not succeed")
    required = os.environ.get("GPU_REQUIRED")
    trusted = os.environ.get("GPU_TRUSTED")
    if required not in ("true", "false") or trusted not in ("true", "false"):
        raise ValueError("missing or invalid gate outputs")
    result = os.environ.get("GPU_RESULT")
    if required == "true":
        if trusted != "true":
            raise ValueError("GPU changes from a fork need a maintainer-controlled branch and PR")
        if result != "success":
            raise ValueError(f"required GPU job did not succeed: {result!r}")
    elif result != "skipped":
        raise ValueError(f"non-GPU change unexpectedly scheduled a GPU job: {result!r}")
    print("GPU validation passed")


def main() -> int:
    try:
        if sys.argv[1:] == ["detect"]:
            detect()
        elif sys.argv[1:] == ["verify"]:
            verify()
        else:
            raise ValueError("usage: gpu_ci_gate.py detect|verify")
    except (KeyError, ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"GPU validation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
