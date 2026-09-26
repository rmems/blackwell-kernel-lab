#!/usr/bin/env python3
"""Detect GPU-relevant changes and fail closed on required GPU job results."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import sys

_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from host_cli import CalledProcessError, capture_git

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


def _cpu_fixture_json(file: Path) -> bool:
    return (
        len(file.parts) >= 3
        and file.parts[0] == "fixtures"
        and file.parts[1] in CPU_FIXTURES
        and file.suffix in {".json", ".jsonl"}
    )


def _device_or_policy_path(path: str, file: Path) -> bool:
    return (
        path in GPU_POLICY
        or file.suffix in DEVICE_SUFFIXES
        or file.name == "CMakeLists.txt"
    )


def needs_gpu(path: str) -> bool:
    """Only known documentation/CPU paths skip; unknown build/source paths run."""
    file = Path(path)
    if file.suffix == ".md" or path == "LICENSE":
        return False
    if _device_or_policy_path(path, file):
        return True
    if path in CPU_FILES or _cpu_fixture_json(file):
        return False
    return True


def commit_sha(value: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", value):
        raise ValueError("missing or invalid event commit SHA")
    return value


def deleted_push(event: dict, event_name: str) -> bool:
    """Branch deletion has no candidate commit to validate or annotate."""
    after = event.get("after")
    return (
        event_name == "push"
        and isinstance(after, str)
        and len(after) in (40, 64)
        and set(after) == {"0"}
    )


def _revision_range(event: dict, event_name: str) -> str:
    if event_name == "pull_request_target":
        pr = event["pull_request"]
        base, head = commit_sha(pr["base"]["sha"]), commit_sha(pr["head"]["sha"])
        return f"{base}...{head}"
    if event_name == "push":
        base, head = commit_sha(event["before"]), commit_sha(event["after"])
        if set(base) == {"0"}:
            raise ValueError("new-branch-requires-gpu")
        return f"{base}..{head}"
    raise ValueError("unsupported event; cannot classify GPU changes")


def changed_paths(event: dict, event_name: str) -> list[str]:
    try:
        revision_range = _revision_range(event, event_name)
    except ValueError as error:
        if str(error) == "new-branch-requires-gpu":
            return ["new-branch-requires-gpu"]
        raise
    # Disabling rename detection preserves both names of a CUDA → docs rename.
    output = capture_git(
        "diff", "--name-only", "--no-renames", "-z", revision_range, "--",
    )
    return [name.decode("utf-8", errors="surrogateescape") for name in output.split(b"\0") if name]


def _assert_repository_identity(event: dict, repository: str) -> None:
    if not repository or event["repository"]["full_name"] != repository:
        raise ValueError("event repository identity mismatch")


def _repository_trusted(event: dict, event_name: str, repository: str) -> bool:
    if event_name != "pull_request_target":
        return True
    return event["pull_request"]["head"]["repo"]["full_name"] == repository


def _gpu_required(event: dict, event_name: str, deleted: bool) -> bool:
    if deleted:
        return False
    if event_name == "workflow_dispatch":
        return True
    if event_name == "push" and event["ref"] != "refs/heads/main":
        return True
    return any(needs_gpu(path) for path in changed_paths(event, event_name))


def _write_gate_outputs(required: bool, trusted: bool, deleted: bool) -> None:
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as stream:
        stream.write(
            f"required={str(required).lower()}\n"
            f"trusted={str(trusted).lower()}\n"
            f"deleted={str(deleted).lower()}\n"
        )


def detect() -> None:
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    event_name = os.environ["GITHUB_EVENT_NAME"]
    repository = os.environ["GITHUB_REPOSITORY"]
    _assert_repository_identity(event, repository)
    trusted = _repository_trusted(event, event_name, repository)
    deleted = deleted_push(event, event_name)
    # Bootstrap pushes must validate the whole candidate, even when the latest
    # pushed range only updates docs after a failed GPU run on the previous head.
    required = _gpu_required(event, event_name, deleted)
    _write_gate_outputs(required, trusted, deleted)
    print(f"GPU required: {required}; trusted repository: {trusted}; branch deleted: {deleted}")


def _read_verify_env() -> tuple[str, str, str]:
    if os.environ.get("DETECTION_RESULT") != "success":
        raise ValueError("change detection did not succeed")
    required = os.environ.get("GPU_REQUIRED")
    trusted = os.environ.get("GPU_TRUSTED")
    if required not in ("true", "false") or trusted not in ("true", "false"):
        raise ValueError("missing or invalid gate outputs")
    return required, trusted, os.environ.get("GPU_RESULT", "")


def _verify_required_gpu(trusted: str, result: str | None) -> None:
    if trusted != "true":
        raise ValueError("GPU changes from a fork need a maintainer-controlled branch and PR")
    if result != "success":
        raise ValueError(f"required GPU job did not succeed: {result!r}")


def _verify_optional_gpu(result: str | None) -> None:
    if result != "skipped":
        raise ValueError(f"non-GPU change unexpectedly scheduled a GPU job: {result!r}")


def verify() -> None:
    required, trusted, result = _read_verify_env()
    if required == "true":
        _verify_required_gpu(trusted, result)
    else:
        _verify_optional_gpu(result)
    print("GPU validation passed")


def main() -> int:
    try:
        if sys.argv[1:] == ["detect"]:
            detect()
        elif sys.argv[1:] == ["verify"]:
            verify()
        else:
            raise ValueError("usage: gpu_ci_gate.py detect|verify")
    except (KeyError, ValueError, OSError, CalledProcessError) as error:
        print(f"GPU validation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
