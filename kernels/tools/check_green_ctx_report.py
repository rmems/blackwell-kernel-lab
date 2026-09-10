#!/usr/bin/env python3
"""Validate a bkl_green_ctx_bench report.

The benchmark's own `ok` sentinel only means it produced a structurally valid
result; it does not mean the 10% decision gate passed, and it does not prove the
JSON is internally consistent. This checker is the part that does: it recomputes
the background wave counts and the per-invocation improvement from the recorded
inputs and requires the report to agree with itself.

Used by .github/workflows/ci-gpu.yml and by the local equivalents in
docs/CI.md, so a local run verifies exactly what CI verifies.

Usage:
    kernels/tools/check_green_ctx_report.py results/green-ctx-bench.json
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

SCHEMA = "bkl.green_ctx_bench.v1"
COMPUTE_CAPABILITY = "12.0"
SAMPLES_PER_MODE = 7
INVOCATIONS = 3

OUTCOMES = {
    "compile_time_api_unavailable",
    "unsupported",
    "not_permitted",
    "no_legal_split",
    "insufficient_vram_headroom",
    "measured",
}


class ReportError(Exception):
    """A report failed validation."""


def require(condition: object, message: str) -> None:
    if not condition:
        raise ReportError(message)


def check_host_headroom(host: dict) -> None:
    require(
        host["projected_free_after_bytes"] >= host["required_headroom_bytes"],
        "projected free VRAM is below the required headroom",
    )
    require(
        host["vram_min_observed_bytes"] >= host["required_headroom_bytes"],
        "minimum observed free VRAM is below the required headroom",
    )


def check_background_waves(capability: dict, workload: dict) -> None:
    per_sm = workload["background_active_blocks_per_sm"]
    blocks = workload["background_grid_blocks"]
    for label, sm_count in (
        ("ordinary", capability["sm_total"]),
        ("partitioned", capability["background_sm_count"]),
    ):
        recomputed = blocks / (per_sm * sm_count)
        reported = workload[f"{label}_background_waves"]
        require(
            math.isclose(reported, recomputed, rel_tol=0.0, abs_tol=1e-6),
            f"{label}_background_waves is {reported}, recomputed {recomputed}",
        )


def check_invocations(invocations: list, required_improvement: float) -> list[bool]:
    require(
        len(invocations) == INVOCATIONS,
        f"expected {INVOCATIONS} invocations, got {len(invocations)}",
    )

    recomputed_gates = []
    for index, invocation in enumerate(invocations, start=1):
        require(invocation["correctness"] is True, f"invocation {index} failed correctness")
        for mode in ("ordinary", "partitioned"):
            samples = invocation[f"{mode}_gpu_latency_ms"]
            require(
                len(samples) == SAMPLES_PER_MODE,
                f"invocation {index} {mode} has {len(samples)} samples,"
                f" expected {SAMPLES_PER_MODE}",
            )
        ordinary = invocation["ordinary_median_ms"]
        recomputed_improvement = (
            100.0 * (ordinary - invocation["partitioned_median_ms"]) / ordinary
        )
        recomputed_gate = recomputed_improvement >= required_improvement
        require(
            invocation["gate_pass"] is recomputed_gate,
            f"invocation {index} reports gate_pass={invocation['gate_pass']},"
            f" recomputed {recomputed_gate} from {recomputed_improvement:.4f}%",
        )
        recomputed_gates.append(recomputed_gate)
    return recomputed_gates


def check_decision(decision: dict, recomputed_gates: list[bool]) -> None:
    passed_all_three = all(recomputed_gates)
    require(
        decision["passed_all_three"] is passed_all_three,
        f"passed_all_three is {decision['passed_all_three']},"
        f" recomputed {passed_all_three}",
    )
    require(
        decision["follow_up_justified"] is passed_all_three,
        f"follow_up_justified is {decision['follow_up_justified']},"
        f" recomputed {passed_all_three}",
    )


def check_measured(data: dict) -> None:
    require(
        data["measurement_skipped_reason"] is None,
        "outcome=measured must not carry a measurement_skipped_reason",
    )

    check_host_headroom(data["host"])

    capability = data["capability"]
    require(
        capability["created_groups"] == 2,
        f"expected 2 Green Context groups, got {capability['created_groups']}",
    )

    check_background_waves(capability, data["workload"])

    summary = data["summary"]
    require(summary["ordinary_median_ms"] > 0, "ordinary median must be positive")
    require(summary["partitioned_median_ms"] > 0, "partitioned median must be positive")

    required_improvement = data["decision"]["required_improvement_percent"]
    recomputed_gates = check_invocations(data["invocations"], required_improvement)
    check_decision(data["decision"], recomputed_gates)


def check_skipped(data: dict) -> None:
    require(
        data["measurement_skipped_reason"],
        f"outcome={data['outcome']} must carry a measurement_skipped_reason",
    )
    require(data["invocations"] == [], "a skipped run must record no invocations")

    summary = data["summary"]
    for field in ("ordinary_median_ms", "partitioned_median_ms", "improvement_percent"):
        require(summary[field] is None, f"a skipped run must leave summary.{field} null")
    require(
        data["decision"]["passed_all_three"] is None,
        "a skipped run must leave decision.passed_all_three null",
    )


def check_report(data: dict) -> None:
    require(data["schema"] == SCHEMA, f"unexpected schema: {data['schema']}")
    require(
        data["host"]["compute_capability"] == COMPUTE_CAPABILITY,
        f"expected compute capability {COMPUTE_CAPABILITY},"
        f" got {data['host']['compute_capability']}",
    )
    require(data["outcome"] in OUTCOMES, f"unknown outcome: {data['outcome']}")

    if data["outcome"] == "measured":
        check_measured(data)
    else:
        check_skipped(data)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {argv[0]} REPORT.json", file=sys.stderr)
        return 2

    path = Path(argv[1])
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        print(f"{path}: {error}", file=sys.stderr)
        return 1

    try:
        check_report(data)
    except (ReportError, KeyError, TypeError, ZeroDivisionError) as error:
        print(f"{path}: {error}", file=sys.stderr)
        return 1

    print(f"{path}: outcome={data['outcome']} ok")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
