#!/usr/bin/env python3
"""CPU contract checks for GPU vs CPU GitHub Actions workflows.

Used by .github/workflows/ci-cpu.yml. Parses workflow YAML as text (no
PyYAML) so GitHub-hosted CPU CI stays dependency-free.

Usage:
    python3 tools/check_gpu_ci_policy.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
SANITIZER_YML = WORKFLOWS / "ci-gpu-sanitizer.yml"
GPU_YML = WORKFLOWS / "ci-gpu.yml"
CPU_YML = WORKFLOWS / "ci-cpu.yml"
LINT_YML = WORKFLOWS / "ci-lint.yml"
RUNNER = ROOT / "kernels" / "tools" / "run_compute_sanitizer.py"
HEADROOM = "kernels/tools/wait_gpu_headroom.sh"
SELF_HOSTED_LABELS = ("self-hosted", "Linux", "X64", "CUDA")
HOSTED = "ubuntu-latest"
CHECKOUT_PIN = "actions/checkout@11d5960a326750d5838078e36cf38b85af677262"
UPLOAD_PIN = "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02"


class PolicyError(Exception):
    """A workflow or runner file violated the CI policy contract."""


def require(condition: object, message: str) -> None:
    if not condition:
        raise PolicyError(message)


def uncommented(text: str) -> str:
    lines = []
    for line in text.splitlines():
        lines.append(line.split("#", 1)[0])
    return "\n".join(lines)


def load(path: Path) -> str:
    require(path.is_file(), f"missing {path.relative_to(ROOT)}")
    return path.read_text()


def event_block(text: str) -> str:
    match = re.search(
        r"^on:\n(.*?)(?=^permissions:|^concurrency:|^jobs:|^env:)",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    require(match, "could not find the workflow on: block")
    return uncommented(match.group(1))


def require_no_self_hosted(path: Path) -> None:
    text = uncommented(load(path))
    require("self-hosted" not in text, f"{path.name} must not schedule self-hosted runners")
    require(f"runs-on: {HOSTED}" in text, f"{path.name} must run on {HOSTED}")


def require_trust_gate(text: str) -> None:
    require(f"runs-on: {HOSTED}" in text, "trust gate must run on GitHub-hosted ubuntu-latest")
    require("head.repo.full_name" in text, "trust gate must compare pull_request.head.repo")
    require("allow=false" in text, "trust gate must skip fork PRs")
    require("persist-credentials: false" in text, "self-hosted checkout must drop credentials")


def require_gpu_runner(text: str) -> None:
    for label in SELF_HOSTED_LABELS:
        require(label in text, f"GPU job must include runner label {label}")
    require(HEADROOM in text, f"GPU job must reuse {HEADROOM}")
    require(CHECKOUT_PIN in text, "checkout action must stay SHA-pinned")


def check_cpu_workflows() -> None:
    require_no_self_hosted(CPU_YML)
    require_no_self_hosted(LINT_YML)
    cpu = load(CPU_YML)
    require(
        "tools/check_gpu_ci_policy.py" in cpu,
        "ci-cpu.yml must run the GPU CI policy checker",
    )
    require(
        "tools/check_compute_sanitizer_runner.py" in cpu,
        "ci-cpu.yml must run the compute-sanitizer runner tests",
    )


def check_gpu_smoke_workflow() -> None:
    text = load(GPU_YML)
    require_trust_gate(text)
    require_gpu_runner(text)
    events = event_block(text)
    require("**/*.md" in events, "ci-gpu.yml must ignore markdown-only cuts")


def check_sanitizer_workflow() -> None:
    text = load(SANITIZER_YML)
    events = event_block(text)
    require("workflow_dispatch:" in events, "sanitizer workflow must be manually dispatchable")
    require("pull_request:" not in events, "sanitizer workflow must not use pull_request")
    require("branches: [main]" in events, "automatic sanitizer runs are main-only")
    require("kernels/src/**" in events, "main-push sanitizer must be path-filtered to kernels")
    require_trust_gate(text)
    require_gpu_runner(text)
    require(UPLOAD_PIN in text, "sanitizer logs must upload with a SHA-pinned artifact action")
    require("if: always()" in text, "sanitizer logs must upload even when the job fails")
    require("run_compute_sanitizer.py --suite" in text, "workflow must call the suite runner")
    require("retention-days: 14" in text, "sanitizer artifacts must be retention-bounded")
    require(CHECKOUT_PIN in text, "sanitizer checkout must stay SHA-pinned")
    require("timeout-minutes: 45" in text, "sanitizer GPU job must be time-bounded")


def check_runner_contract() -> None:
    text = load(RUNNER)
    require("--error-exitcode" in text, "runner must pass --error-exitcode so findings fail CI")
    require("ERROR SUMMARY" in text, "runner must parse Compute Sanitizer ERROR SUMMARY")
    require("bkl_device_hello" in text, "suite must cover device_hello")
    require("bkl_graph_launch_bench" in text, "suite must cover graph-launch")
    require("bkl_green_ctx_bench" in text, "suite must cover green-ctx")
    require("--smoke" in text, "bench sanitizer jobs must use --smoke")
    require("initcheck" in text, "suite must include an initcheck justified by global memory")
    require("sha256" in text, "runner must record a binary digest")
    require("sanitizer_version" in text, "runner must record sanitizer version")
    src = ROOT / "kernels" / "src"
    for path in sorted(src.glob("*.cu")):
        source = path.read_text()
        require(
            "bkl_oob_example" not in source,
            f"{path.name} must not ship the documented intentional OOB example",
        )
        require(
            "BKL_INTENTIONAL_OOB" not in source,
            f"{path.name} must not ship an intentional sanitizer fault",
        )


def main() -> int:
    try:
        check_cpu_workflows()
        check_gpu_smoke_workflow()
        check_sanitizer_workflow()
        check_runner_contract()
    except PolicyError as error:
        print(error, file=sys.stderr)
        return 1
    print("gpu ci policy: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
