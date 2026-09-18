"""Markdown / JSON writers for F0 efficiency summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def fmt_num(value: Any, unit: str = "", missing: str = "n/a") -> str:
    if value is None:
        return missing
    if unit:
        return f"{value} {unit}"
    return str(value)


def report_header(summary: dict[str, Any]) -> list[str]:
    run = summary["run"]
    return [
        f"# F0 efficiency summary (`{summary['schema_version']}`)",
        "",
        f"- Run: `{run.get('agoge_run_id')}`",
        f"- Model: `{run.get('model_id')}` (formulas are model-agnostic)",
        f"- Host: `{((run.get('host') or {}).get('hostname'))}`",
        "",
    ]


def report_throughput(summary: dict[str, Any]) -> list[str]:
    thr = summary["throughput"]
    step = summary["step_time"]
    return [
        "## Throughput (Agoge counters + monotonic_ns)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| tokens/s | {fmt_num(thr.get('tokens_per_s'), 'token/s')} |",
        f"| examples/s | {fmt_num(thr.get('examples_per_s'), 'example/s')} |",
        f"| elapsed | {fmt_num(thr.get('elapsed_s'), 's')} |",
        f"| tokens delta | {fmt_num(thr.get('tokens_accepted_delta'))} |",
        f"| step-time n / mean / stdev | {step['count']} / {fmt_num(step.get('mean'), 's')} / {fmt_num(step.get('stdev'), 's')} |",
        "",
        "Tokens/s and step-time are **not** inferred from GPU utilization.",
        "",
    ]


def report_vram(summary: dict[str, Any]) -> list[str]:
    vram = summary["vram"]
    fit = summary["train_fit_comparison"]
    return [
        "## VRAM / headroom (BKL samples)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| peak used | {fmt_num(vram.get('peak_used_mib'), 'MiB')} |",
        f"| min free | {fmt_num(vram.get('min_free_mib'), 'MiB')} |",
        f"| min headroom | {fmt_num(vram.get('min_headroom_mib'), 'MiB')} |",
        f"| host floor | {vram.get('headroom_floor_mib')} MiB (meets={vram.get('meets_host_floor')}) |",
        f"| ok / missing used samples | {vram.get('ok_count')} / {vram.get('missing_count')} |",
        "",
        f"Concise train-fit SoT remains **{fit['source_of_truth']}**. "
        "This summary is correlated evidence, not a second ceiling.",
        "",
    ]


def report_power_energy(summary: dict[str, Any]) -> list[str]:
    cov = summary["coverage"]
    power = summary["power"]
    temp = summary["temperature_gpu"]
    energy = summary["energy"]
    return [
        "## Power / thermal",
        "",
        "| Metric | Value | missing |",
        "|---|---|---|",
        f"| board power avg / peak | {fmt_num(power.get('avg_w'), 'W')} / {fmt_num(power.get('peak_w'), 'W')} | {power.get('missing_count')} |",
        f"| GPU temp peak | {fmt_num(temp.get('peak_c'), 'C')} | {temp.get('missing_count')} |",
        "",
        "## Energy (approximate)",
        "",
        energy["note"],
        "",
        "| Item | Value |",
        "|---|---|",
        f"| method | {energy.get('method')} |",
        f"| planned cadence | {fmt_num(cov.get('planned_cadence_ms'), 'ms')} |",
        f"| max gap factor | {energy.get('max_gap_factor')} × cadence |",
        f"| pairs integrated / long-gap / missing-power | {energy.get('pairs_integrated')} / {energy.get('pairs_skipped_long_gap')} / {energy.get('pairs_skipped_missing_power')} |",
        f"| approximate joules | {fmt_num(energy.get('approximate_joules'), 'J')} |",
        f"| approximate J/step | {fmt_num(energy.get('approximate_joules_per_step'), 'J/step')} |",
        f"| approximate J/token | {fmt_num(energy.get('approximate_joules_per_token'), 'J/token')} |",
        "",
        "## Coverage",
        "",
        f"- samples {cov.get('sample_count')}, expected {fmt_num(cov.get('expected_samples'))}, "
        f"coverage_ratio {fmt_num(cov.get('coverage_ratio'))}",
        f"- joined {cov.get('joined_count')}, unjoined {cov.get('unjoined_count')}, long gaps {cov.get('long_gap_count')}",
        f"- policy: {cov.get('missing_sample_policy')}",
        "",
    ]


def report_phases(summary: dict[str, Any]) -> list[str]:
    lines = ["## By phase (joined samples only)", ""]
    by_phase = summary.get("by_phase") or {}
    if not by_phase:
        lines.append("_No joined samples; phases are not inferred from GPU load._")
        lines.append("")
        return lines
    for name in sorted(by_phase):
        block = by_phase[name]
        gpu = block["utilization_gpu"]
        mem = block["utilization_memory"]
        lines.append(
            f"- `{name}`: n={block['sample_count']}; GPU util avg {fmt_num(gpu.get('avg'), '%')}; "
            f"mem util avg {fmt_num(mem.get('avg'), '%')}; "
            f"peak VRAM {fmt_num(block['vram'].get('peak_used_mib'), 'MiB')}"
        )
    return lines


def report_profiles(summary: dict[str, Any]) -> list[str]:
    windows = summary.get("profile_windows") or []
    if not windows:
        return []
    lines = ["", "## Profiler windows", ""]
    for window in windows:
        compact = window.get("compact_summary") or {}
        lines.append(
            f"- `{window['profile_window_ref']}`: samples={window['sample_count']}, "
            f"phases={window['joined_phases']}, runtime_ms={compact.get('runtime_ms')}"
        )
    return lines


def report_lines(summary: dict[str, Any]) -> list[str]:
    lines = report_header(summary)
    lines.extend(report_throughput(summary))
    lines.extend(report_vram(summary))
    lines.extend(report_power_energy(summary))
    lines.extend(report_phases(summary))
    lines.extend(report_profiles(summary))
    return lines


def write_json(path: Path | None, summary: dict[str, Any]) -> str:
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return text


def write_report(path: Path | None, summary: dict[str, Any]) -> str:
    text = "\n".join(report_lines(summary)) + "\n"
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return text
