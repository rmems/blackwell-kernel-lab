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

JSONL, one JSON object per line. Two kinds share an envelope and differ in
payload ownership:

| `record_kind` | Owner | File in this repo (fixture) |
|---|---|---|
| `agoge_marker` | **agoge-forger** | `fixtures/f0-correlation/agoge-markers.jsonl` |
| `bkl_gpu_sample` | **this repo** | `fixtures/f0-correlation/bkl-gpu-samples.jsonl` |

BKL **must not** infer `phase`, `global_step`, or other Agoge semantics from
GPU load, clocks, or power. If a marker is missing, the join leaves Agoge
fields unset; it does not guess them.

There is no permanent collector process in this contract. A train wrapper
(#53) and Agoge (#144) may write append-only JSONL for the life of one run
and exit.

## Envelope (both kinds)

| Field | Type | Owner of the *value* | Notes |
|---|---|---|---|
| `schema_version` | string | contract | Must be `bkl.f0_correlation.v1` |
| `record_kind` | string | contract | `agoge_marker` or `bkl_gpu_sample` |
| `agoge_run_id` | string | **Agoge** | Stable id for one train invocation. Join key. |
| `host` | object | capturing process | `hostname` (string). Same-host capture assumed. |
| `gpu` | object | capturing process | Identity only: `pci_bus_id` and/or `uuid`, `name`, `compute_capability` (`"12.0"` on this 5080). |
| `timestamp_utc` | string | capturing process | RFC 3339 UTC with `Z`. Wall clock. |
| `monotonic_ns` | integer ≥ 0 | capturing process | `CLOCK_MONOTONIC` nanoseconds. Local order. |
| `collector` | object | capturing process | `id` (e.g. `agoge-forger`, `bkl-gpu-sampler`), `version` (string). |
| `cadence_ms` | integer or null | BKL samples | Planned sample interval. Markers use `null`. |

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

Designed for #53 (sampler) and #54 (profile window refs). This PR does not
sample the 5080.

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
| `headroom` | `MiB` | Raw free-vs-total if sampled. Peak used / min headroom vs the 2 GiB floor are derived in [#55](https://github.com/rmems/blackwell-kernel-lab/issues/55) ([`f0-efficiency-metrics.md`](f0-efficiency-metrics.md)) |
| `throttle` | object | `reasons` (list of strings) when `status` is `ok`; else `status` + empty/`null` reasons — see fixture |
| `profile_window_ref` | string or null | Opaque id for a #54 CUDA window; not a `.cu` dump |
| `cuda` | object | `driver_version`, `runtime_version`, `tool` (strings; unknown → `null`) |

Do not require every backend to fill every metric. Use `unsupported` /
`unavailable`.

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
5. Wall clock is for humans and for skew notes. Ordering is monotonic.
   Same-host capture: assume NTP/chrony keeps UTC within **1 s**; do not
   invent a distributed trace. Cross-machine join is a non-goal.

Clock domains: `monotonic_ns` is comparable only across processes on the same
boot of the same host. The fixture uses one synthetic domain.

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
  --samples fixtures/f0-correlation/bkl-gpu-samples.jsonl
```

`ci-cpu` runs the same command.

Derived tokens/s, step-time, peak VRAM, headroom, and approximate energy:
[`f0-efficiency-metrics.md`](f0-efficiency-metrics.md)
(`tools/summarize_f0_efficiency.py`, #55 / RM-1232).
