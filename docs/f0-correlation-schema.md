# F0 correlation schema (`bkl.f0_correlation.v1`)

**Issue:** [#52](https://github.com/rmems/blackwell-kernel-lab/issues/52) ·
Linear twin RM-1229 (when the mirror is reachable).

**Audience:** `agoge-forger` (emit markers) and this repo (emit GPU samples).
Join is a CPU contract. It does not occupy the RTX 5080.

This is **not** a trainer, a daemon, or a kernel. Schema lives here so both
repos share one versioned join key. Training orchestration stays in
[`agoge-forger`](https://github.com/rmems/agoge-forger) — see
[`FORGE_BOUNDARY.md`](FORGE_BOUNDARY.md). Agoge issue
[#144](https://github.com/rmems/agoge-forger/issues/144) should emit the
Agoge-owned records described below; this lab does not implement that trainer
path.

## Version

| Field | Value |
|---|---|
| `schema_version` | `bkl.f0_correlation.v1` |

Bump the suffix (`v2`, …) when a required field is renamed, a unit changes,
or ownership of a field moves. Additive optional fields do **not** require a
bump; consumers must ignore unknown keys (forward compatibility).

## Record kinds

JSONL, one JSON object per line. Marker and sample kinds share an envelope
and differ in payload ownership. `bkl_clock_skew` uses the same envelope and
is a per-run sidecar (not mixed into the marker or sample JSONL streams):

| `record_kind` | Owner | File in this repo (fixture) |
|---|---|---|
| `agoge_marker` | **agoge-forger** | `fixtures/f0-correlation/agoge-markers.jsonl` |
| `bkl_gpu_sample` | **this repo** | `fixtures/f0-correlation/bkl-gpu-samples.jsonl` |
| `bkl_clock_skew` | **this repo** | `fixtures/f0-correlation/clock-skew.json` |

BKL **must not** infer `phase`, `global_step`, or other Agoge semantics from
GPU load, clocks, or power. If a marker is missing, the join leaves Agoge
fields unset; it does not guess them.

There is no permanent collector process in this contract. A train wrapper
(#53) and Agoge (#144) may write append-only JSONL for the life of one run
and exit.

## Envelope (all kinds)

| Field | Type | Owner of the *value* | Notes |
|---|---|---|---|
| `schema_version` | string | contract | Must be `bkl.f0_correlation.v1` |
| `record_kind` | string | contract | `agoge_marker`, `bkl_gpu_sample`, or `bkl_clock_skew` |
| `agoge_run_id` | string | **Agoge** | Stable id for one train invocation. Join key. |
| `host` | object | capturing process | `hostname` (string). Same-host capture assumed. |
| `gpu` | object | capturing process | Identity only: `pci_bus_id` and/or `uuid`, `name`, `compute_capability` (`"12.0"` on this 5080). |
| `timestamp_utc` | string | capturing process | RFC 3339 UTC with `Z`. Wall clock. |
| `monotonic_ns` | integer ≥ 0 | capturing process | `CLOCK_MONOTONIC` nanoseconds. Local order. |
| `collector` | object | capturing process | `id` (e.g. `agoge-forger`, `bkl-gpu-sampler`, `bkl-clock-calibrator`), `version` (string). |
| `cadence_ms` | integer or null | BKL samples | Planned sample interval. Markers and clock-skew reports use `null`. |

`gpu` identity is copied onto Agoge markers when Agoge knows the device so
joins can assert one card. BKL samples always fill it. Missing identity is
`null` on the object fields that are unknown — not an empty string treated as
a device.

## Physical measurements (missing ≠ zero)

Every physical quantity is an object, never a bare number:

```json
{ "value": 42.0, "unit": "W", "status": "ok" }
```

| `status` | `value` | Meaning |
|---|---|---|
| `ok` | number (may be `0`) | Measurement was taken. Zero is a real zero. |
| `unavailable` | `null` | Backend exists but this sample has no reading. |
| `unsupported` | `null` | This host/tool cannot expose the field. |
| `permission_denied` | `null` | NVML returned `NO_PERMISSION` for this field. |
| `transient_failure` | `null` | Timeout / not-ready / unknown backend error. |
| `device_lost` | `null` | GPU handle lost (`NVML_ERROR_GPU_IS_LOST` / not found). |

These extra missingness statuses are additive. They do not bump
`bkl.f0_correlation.v1`. Capability discovery (what the device can expose
before #53 samples) lives in
[`f0-nvml-capability.md`](f0-nvml-capability.md) (`bkl.f0_capability.v1`).

Encoding “no power reading” as `0` is a schema violation. The CPU validator
rejects `status != ok` with a non-null `value`, and rejects `status == ok`
with `value == null`.

## Agoge-owned payload (`agoge_marker`)

Emitted by the trainer. BKL only stores and joins these fields.

| Field | Type | Unit / notes |
|---|---|---|
| `model_id` | string | Hugging Face / Agoge model id |
| `model_revision` | string | Immutable revision (git SHA or snapshot id) |
| `dataset` | object | `id`, `split`, `config_digest` (strings; digest may be `null` if Agoge has none yet) |
| `phase` | string | Trainer phase (`train`, `eval`, …). **Not** inferred from GPU util. |
| `global_step` | integer ≥ 0 | Optimizer / Agoge global step |
| `microstep` | integer ≥ 0 or null | Optional inner step |
| `tokens_accepted` | integer ≥ 0 or null | Agoge counter |
| `examples_accepted` | integer ≥ 0 or null | Agoge counter |
| `loss` | measurement or omit | Unit `1` (dimensionless). Optional. |
| `throughput_tokens_per_s` | measurement or omit | Unit `token/s`. Optional; Agoge-owned, not derived here. |

## BKL-owned payload (`bkl_gpu_sample`)

Designed for #53 (sampler) and #54 (profile window refs). Capability discovery
for these fields is [`f0-nvml-capability.md`](f0-nvml-capability.md) (RM-1350).

| Field | Unit | Notes |
|---|---|---|
| `power` | `W` | Instantaneous board power if the tool exposes it |
| `temperature_gpu` | `C` | GPU die |
| `temperature_memory` | `C` | Memory, if exposed |
| `utilization_gpu` | `%` | 0–100 when `ok` |
| `utilization_memory` | `%` | 0–100 when `ok` |
| `clock_graphics` | `MHz` | |
| `clock_memory` | `MHz` | |
| `vram_used` | `MiB` | |
| `vram_free` | `MiB` | |
| `vram_total` | `MiB` | |
| `headroom` | `MiB` | Free relative to the 2 GiB host floor is a later #55 derivation; this field is raw free-vs-total if sampled |
| `throttle` | object | `reasons` (list of strings) when `status` is `ok`; else `status` + empty/`null` reasons — see fixture |
| `profile_window_ref` | string or null | Opaque id for a #54 CUDA window; not a `.cu` dump |
| `cuda` | object | `driver_version`, `runtime_version`, `tool` (strings; unknown → `null`) |
| `capability_digest` | string or omit | Optional `sha256:<64 hex>` binding this sample to the run’s `bkl_gpu_capability` snapshot. Required when validating a bound run. |

Do not require every backend to fill every metric. Use `unsupported` /
`unavailable` / `permission_denied` / `transient_failure` / `device_lost`.

`throttle` shape when present:

```json
{ "status": "ok", "reasons": [] }
```

or `{ "status": "unsupported", "reasons": null }`.

## Join rules (one host)

Deterministic join used by `tools/check_f0_correlation.py`:

1. Partition records by `agoge_run_id`. Different run ids never join.
2. Within a run, sort Agoge markers by `monotonic_ns` (tie-break `timestamp_utc`).
3. For each BKL sample, attach the latest marker with
   `marker.monotonic_ns <= sample.monotonic_ns` on the **same** `host.hostname`
   **and** the same GPU identity (`uuid` if both sides have a non-empty uuid;
   otherwise `pci_bus_id` if both sides have a non-empty bus id). Mismatched
   or missing identities stay unjoined.
4. If no such marker exists, the sample is **unjoined** (allowed in production
   streams; the CPU fixture in this repo requires ≥1 joined pair).
5. Wall clock is for humans and for the skew report below. Ordering is
   monotonic. Same-host capture: assume NTP/chrony keeps UTC within **1 s**;
   do not invent a distributed trace. Cross-machine join is a non-goal.
6. After the monotonic candidate is chosen, apply the clock-skew gate:
   pairs that straddle a **backward** wall-clock jump or a monotonic
   regression are **refused**; pairs that straddle a **forward** jump larger
   than the 1 s bound are kept but marked **degraded**. Pairs that stay on
   one side of a discontinuity still join.

Clock domains: `monotonic_ns` is comparable only across processes on the same
boot of the same host. The fixture uses one synthetic domain.

## Clock calibration (one-host skew report)

`bkl_clock_skew` is the timestamp bridge that lets Agoge markers, BKL GPU
samples, and later #54 profiling windows share one local timebase. It is
**not** NTP management, not a daemon, and not a distributed clock.

Capture paired UTC wall-clock and `CLOCK_MONOTONIC` observations at run
**start** and **end** (`tools/f0_clock.py:capture_observation`). From those
pairs the calibrator records:

| Field | Meaning |
|---|---|
| `observations` / `start_observation` / `end_observation` | Paired `timestamp_utc` + `monotonic_ns` (+ derived `wall_ns`) |
| `offset_ns` | `wall_ns - monotonic_ns` at start |
| `end_offset_ns` | Same at end, or `null` if the end observation is missing |
| `drift_ns_per_s` | `(end_offset - start_offset)` per monotonic second; `null` without an end pair |
| `clock_resolution` | `monotonic_ns` and `realtime_ns` (fixture uses 1 ns; live capture may read `clock_getres`) |
| `assumed_utc_bound_ns` | 1_000_000_000 (the 1 s same-host bound) |
| `discontinuities` | Detected jumps, each with `kind` and `correlation` |
| `validity` | `ok`, `degraded`, or `refused` |
| `join_order` | Always `monotonic` |
| `correlation_across_discontinuity` | `allowed`, `degraded`, or `refused` |

Envelope matches the other kinds (`schema_version`, `agoge_run_id`, `host`,
`gpu`, `collector`, start `timestamp_utc` / `monotonic_ns`). `cadence_ms`
is `null`. Additive optional kind: bump is not required; unknown keys stay
ignored.

### Discontinuity kinds

| `kind` | Detection | Join across it |
|---|---|---|
| `backward_wall_clock` | Wall delta `< 0`, or wall lagged monotonic by more than the 1 s bound | **refused** |
| `forward_jump` | Wall jumped ahead of monotonic by more than the 1 s bound | **degraded** |
| `monotonic_regression` | `monotonic_ns` went backwards (should not happen for `CLOCK_MONOTONIC`) | **refused** |

Missing the end observation is **degraded** (offset is still known; drift is
null). Sparse markers are not a clock fault: join still uses the latest
preceding marker by monotonic time.

### Assumptions and limits

- One host, one boot. `monotonic_ns` is not comparable after reboot or across
  machines.
- UTC is best-effort. Chrony/NTP may slew or step; this report detects the
  step, it does not correct the host clock.
- Python `datetime` stores microseconds. Live `time.time_ns()` pairs keep
  integer `wall_ns` for exact offset; RFC 3339 is the portable form.
- No training-phase inference from GPU load. This file only calibrates time.
- CPU-only. It does not run on the RTX 5080 and does not consume the 16 GB
  VRAM budget; leave ≥2 GiB free when a later #53/#54 GPU capture runs.

Scenario fixtures: `fixtures/f0-correlation/clock-skew/` (`stable`,
`bounded_drift`, `backward_jump`, `forward_jump`, `sparse_markers`,
`missing_end`).

## What Agoge should emit (#144)

Append-only JSONL, one `agoge_marker` per phase/step boundary (and optionally
on a coarse timer). Required envelope + Agoge payload. Do not wait for BKL
to be running. Include `agoge_run_id` in the train CLI so #53 can stamp the
same id on GPU samples even if marker emission lands later.

## Mapping later (Theseus / Thalamic)

Field names here can be copied or aliased into a Machine Physiology /
Thalamic envelope later. This schema does **not** claim BKL owns Thalamic’s
safety contract.

## Hardware note

RTX 5080, `sm_120`, ~16 GB. Leave ≥2 GiB free when GPU work runs. Validating
this schema is CPU-only.

## Validate

```bash
python3 tools/check_f0_correlation.py \
  --markers fixtures/f0-correlation/agoge-markers.jsonl \
  --samples fixtures/f0-correlation/bkl-gpu-samples.jsonl \
  --clock-skew fixtures/f0-correlation/clock-skew.json
python3 tools/check_f0_clock_skew.py \
  --fixtures fixtures/f0-correlation/clock-skew
```

`ci-cpu` runs the same commands.

Capability snapshot (RM-1350, CPU):

```bash
python3 tools/check_nvml_capability.py
```
