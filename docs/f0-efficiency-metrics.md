# F0 derived efficiency and headroom metrics (`bkl.f0_efficiency.v1`)

**Issue:** [#55](https://github.com/rmems/blackwell-kernel-lab/issues/55) ·
Linear twin RM-1232.

**Audience:** humans and later analysis jobs that consume a correlated
Agoge ↔ BKL bundle. This is **not** a trainer, a hyperparameter tuner, or a
thermal/power safety controller (`thalamic-relay` owns hard-safety).

Join remains the CPU contract in
[`f0-correlation-schema.md`](f0-correlation-schema.md)
(`bkl.f0_correlation.v1`). This page only **derives** measurements from that
bundle. RTX 5080 / `sm_120` / ~16 GB; leave **≥2 GiB** free. Summarizing a
bundle is CPU-only and does not occupy the card.

## Inputs

| Input | Owner | Required |
|---|---|---|
| `agoge_marker` JSONL | agoge-forger | for tokens/s, examples/s, step-time, phase |
| `bkl_gpu_sample` JSONL | this repo (#53) | for VRAM, power, util, temperature, energy |
| optional compact profile JSONL | this repo (#54) | kernel/runtime fields on `profile_window_ref` |
| optional train-fit reference JSON | this repo (#44) | comparison only; never overwrites the table |

BKL **must not** infer `phase`, `global_step`, tokens/s, or step-time from GPU
load, clocks, or power. If a marker is missing, those Agoge fields stay unset.

The same formulas apply to MiniCPM5, Granite, or any later Agoge model id.
There is no per-model branch.

## Units

| Quantity | Unit | Source |
|---|---|---|
| time | `s` (from `monotonic_ns`) | Agoge markers for step/throughput; BKL samples for energy/coverage |
| tokens/s, examples/s | `token/s`, `example/s` | Agoge counters ÷ Agoge elapsed time |
| VRAM used / free / headroom | `MiB` | ok `bkl_gpu_sample` extrema |
| board power | `W` | ok samples |
| temperature | `C` | ok `temperature_gpu` |
| utilization | `%` | ok samples |
| energy | `J` (approximate) | trapezoid of board power over time |
| J/step, J/token | `J/step`, `J/token` (approximate) | energy ÷ Agoge step count or token delta |

Physical fields keep the #52 rule: **missing ≠ zero**. `unavailable` /
`unsupported` / `invalid` never enter min/max/mean and never become `0 W` or
`0 MiB`.

## Aggregation rules

Every numeric block records `ok_count`, `unavailable_count`,
`unsupported_count`, `invalid_count`, and `missing_count` (the last three
summed). Extrema and means use **ok samples only**.

| Metric | Rule |
|---|---|
| tokens/s | `(tokens_last − tokens_first) / ((t_last − t_first)/1e9)` on Agoge markers sorted by `monotonic_ns`. Requires both counters, `tokens_last ≥ tokens_first`, and `Δt > 0`. |
| examples/s | Same with `examples_accepted`. |
| Agoge-reported `throughput_tokens_per_s` | Copied from the last marker if present. **Not** the derived tokens/s. |
| step-time | `Δmonotonic_ns / 1e9` between consecutive markers whose `global_step` **increases**. Report count, min, max, mean, sample variance, stdev. Variance/stdev are `null` when `n < 2`. |
| peak VRAM used | `max(vram_used)` among ok samples |
| min VRAM free | `min(vram_free)` among ok samples |
| min headroom | `min(headroom)` among ok samples |
| effective min headroom | Per sample: ok `headroom`, else ok `vram_free`. Then `min` of those coalesced values. A later missing-headroom / low-free sample can fail the floor. |
| host floor | `effective_min_headroom ≥ 2048 MiB` (the existing 2 GiB rule). `null` if no coalesced values. |
| device filter | Physical metrics use samples whose `host.hostname` and GPU identity match the first marker (or first sample). Other-device rows are counted in `excluded_foreign_sample_count` and are not folded into VRAM/energy/util. |
| avg / peak board power | mean / max of ok `power` |
| peak GPU temperature | max of ok `temperature_gpu` |
| avg GPU / memory util by phase | join samples to the latest marker (`f0-correlation-schema.md`); group by `marker.phase`; mean of ok util in that group. Unjoined samples are **not** assigned a guessed phase. |

Peak VRAM / min headroom in a summary **must** equal those extrema on the
underlying ok samples (integer MiB in the fixtures). That is the aggregation
semantics #44 compares against.

## Energy (approximate)

Energy is an **estimate** obtained by integrating sampled **board power**
over time. It is not wall-power, not a calorimeter, and not a datacenter
PUE number.

| Item | Policy |
|---|---|
| Method | Trapezoid: for consecutive samples i→i+1, `J += 0.5 × (P_i + P_{i+1}) × Δt` with `P` in watts and `Δt` in seconds from `monotonic_ns`. |
| Cadence | Each sample carries planned `cadence_ms`. The report lists the unique planned cadences and the observed median interval. |
| Max gap | Default `max_gap_factor = 3`. If `Δt > 3 × min(cadence_i, cadence_{i+1})`, **do not interpolate**. Count the pair as a long gap. |
| No cadence | If either sample lacks a positive cadence, treat the pair as unbounded and skip. |
| Missing power | If either `power.status != ok`, skip the pair. Do not substitute `0 W`. A pair that is also longer than the allowed cadence is recorded as a long gap **and** a missing-power skip. |
| Result | If no pair integrates, `approximate_joules` is `null` (missing ≠ `0 J`). |
| J/step | `approximate_joules / n` only when the energy window matches the Agoge counter window: no long-gap or missing-power skips, every sample inside the marker range, and `integrated_span_s` equals marker `elapsed_s`. Otherwise `null` with `rates_omitted_reason: energy_window_mismatch`. |
| J/token | Same alignment rule, then `approximate_joules / tokens_accepted_delta`. |

The JSON and the human report both set `energy_is_approximate: true` and
repeat this caveat.

## Coverage

```text
expected_samples = 1 + floor((t_last − t_first) / median_planned_cadence)
coverage_ratio   = observed_samples / expected_samples
# median_planned_cadence keeps a fractional median (do not truncate to int)
```

A single sample has `expected_samples = 1`. If cadence is missing, expected
count and ratio are `null`. Long gaps are listed with `dt_ms` and
`max_allowed_ms`; they lower energy coverage without inventing samples.

## Profiler windows (#54)

`bkl_gpu_sample.profile_window_ref` is an opaque join key, not a `.cu` dump.
The summarizer groups samples by that ref and, when `--profile-windows` is
given, attaches the matching compact record (`runtime_ms`, `kernel_count`,
top kernels, H2D/D2H, sync stalls). Kernel families are **not** guessed from
utilization. Ordinary Agoge runs do not require a profiler.

## Train-fit comparison (#44)

Issue [#44](https://github.com/rmems/blackwell-kernel-lab/issues/44)
(RM-1053 / `docs/TRAIN_FIT_5080.md` when that page is on the default branch)
remains the **concise train-fit summary** for this 16 GB host. This
derivation is synchronized evidence, not a second ceiling.

The 2026-09-11 MiniCPM5 canary peak (**2.49 GiB** trainer
`max_memory_allocated`) is a different measurement basis than sampler
`vram_used`. The summarizer therefore reports `comparable_peak: false` unless
the optional `--train-fit-ref` JSON sets `peak_basis` to
`bkl_gpu_sample.vram_used`. Headroom comparison on
`bkl_gpu_sample.vram_free` uses derived `min_free_mib`, not
`effective_min_headroom_mib`. Disagreement across bases is **not** a license
to rewrite the #44 table.

The correlation fixture (`run_minicpm5_fixture_001`) is synthetic join data
for CPU tests. It is not the 2026-09-11 canary row.

## Output

Machine-readable JSON (`schema_version: bkl.f0_efficiency.v1`) and a markdown
table. Example from the MiniCPM5 correlation fixture:
`fixtures/f0-efficiency/minicpm5-summary.json`.

```bash
python3 tools/summarize_f0_efficiency.py \
  --markers fixtures/f0-correlation/agoge-markers.jsonl \
  --samples fixtures/f0-correlation/bkl-gpu-samples.jsonl \
  --profile-windows fixtures/f0-efficiency/profile-windows.jsonl \
  --train-fit-ref fixtures/f0-efficiency/train-fit-reference.json \
  --out results/f0-efficiency.json \
  --report results/f0-efficiency.md

python3 tools/check_f0_efficiency.py
```

`ci-cpu` runs the checker (formulas + MiniCPM5 fixture + Granite-model-id
sanity). Write live run output under gitignored `results/`.

## Non-goals

- Datacenter-grade energy claims from board-power sampling
- Automatic hyperparameter tuning
- Thermal/power safety (not `thalamic-relay`)
- Cross-GPU ranking until equivalent bundles exist on other hardware
