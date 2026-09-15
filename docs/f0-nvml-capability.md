# F0 NVML capability snapshot (`bkl.f0_capability.v1`)

**Issue:** Linear RM-1350 · GitHub [#52](https://github.com/rmems/blackwell-kernel-lab/issues/52)
schema (parent) · prerequisite for [#53](https://github.com/rmems/blackwell-kernel-lab/issues/53)
sampler.

**Audience:** this repo (discover before sampling) and `agoge-forger` (consume
the bound digest). Discovery is a CPU contract plus an optional one-shot NVML
probe. It does not occupy the RTX 5080 in CI.

This is **not** the #53 sampler loop, a daemon, a safety controller, or a
cross-vendor GPU abstraction. Unsupported sensors stay `unsupported`; they are
not estimated.

## Version

| Field | Value |
|---|---|
| `schema_version` | `bkl.f0_capability.v1` |
| `record_kind` | `bkl_gpu_capability` |

Bump the suffix when a required field is renamed, a unit changes, or ownership
of a field moves. Additive optional fields do not require a bump.

Measurement `status` values are shared with
[`f0-correlation-schema.md`](f0-correlation-schema.md) (`bkl.f0_correlation.v1`).

## Why this exists

The F0 sampler must know which NVML fields this RTX 5080 actually exposes
**before** it starts writing `bkl_gpu_sample` records. A missing power reading
must never become `0 W`. Idle `0%` utilization is a real zero and must stay
`status: ok`.

One capability snapshot is bound to one `agoge_run_id`. GPU samples on that run
copy `capability_digest`.

## Missingness

Every physical quantity is an object (`value`, `unit`, `status`), never a bare
number. Capability snapshots also store `capability` next to `status`.

| `status` | `value` | `capability` | Meaning |
|---|---|---|---|
| `ok` | number (may be `0`) | `supported` | Probe succeeded. Zero is a real zero. |
| `unavailable` | `null` | `supported` | Sensor exists; this probe has no reading (`NVML_ERROR_NO_DATA`). |
| `unsupported` | `null` | `unsupported` | This host/tool cannot expose the field. |
| `permission_denied` | `null` | `permission_denied` | NVML returned `NO_PERMISSION`. |
| `transient_failure` | `null` | `supported` | Timeout / not ready / unknown NVML error. Retry later. |
| `device_lost` | `null` | `device_lost` | GPU lost or NVML handle gone. |

`throttle` uses the same `status` set with `reasons` (list when `ok`, otherwise
`null`) and `value`/`unit` always `null`.

Schema validation rejects `status != ok` with a non-null `value`, including
fabricated `0`. It also rejects `status == ok` with `value == null`.

`device_status` on the snapshot (not a metric) is one of: `ok`, `no_device`,
`device_lost`, `permission_denied`, `transient_failure`, `unsupported`.
`NVML_ERROR_DRIVER_NOT_LOADED` is `unsupported` (no driver), not `device_lost`.

## Snapshot fields

Envelope matches the correlation records so a run can carry both files:

| Field | Notes |
|---|---|
| `agoge_run_id` | Same join key as GPU samples. |
| `host.hostname` | Same-host capture assumed. |
| `gpu` | `uuid`, `pci_bus_id`, `name`, `compute_capability`. Null when unknown; empty string is invalid. `device_status=ok` requires uuid or PCI. |
| `tools` | `nvml_version`, `driver_version`, `nvidia_smi`. Live probes fill NVML/driver from libnvidia-ml and leave `nvidia_smi` null (nvidia-smi is **not** invoked and is **not** a metric source). Unknown → `null`. |
| `collector` | `id` = `bkl-nvml-capability`, `version` string. |
| `metrics` | One object per F0 field below. |
| `capability_digest` | `sha256:` + 64 lowercase hex over canonical JSON of `schema_version`, `device_status`, `gpu`, `tools`, and per-metric `capability`+`unit` (not probe values). |
| `metric_notes` | Optional NVML detail strings. Never a substitute for `status`. |

### Metrics probed

| Metric | Unit | Source (live NVML) |
|---|---|---|
| `power` | `W` | `nvmlDeviceGetPowerUsage` (mW → W) |
| `temperature_gpu` | `C` | `nvmlDeviceGetTemperature` (GPU) |
| `temperature_memory` | `C` | memory temp API or sensor id 1; often `unsupported` |
| `utilization_gpu` | `%` | `nvmlDeviceGetUtilizationRates` |
| `utilization_memory` | `%` | same struct |
| `clock_graphics` | `MHz` | `nvmlDeviceGetClockInfo` graphics |
| `clock_memory` | `MHz` | memory clock |
| `vram_used` / `vram_free` / `vram_total` | `MiB` | `nvmlDeviceGetMemoryInfo` |
| `headroom` | `MiB` | copy of free from that same call (not a 2 GiB-floor derivation; #55) |
| `throttle` | reasons | `nvmlDeviceGetCurrentClocksThrottleReasons` |
| `performance_state` | `pstate` | `nvmlDeviceGetPerformanceState` (`0` = P0) |

Optional metric failure does **not** abort the remaining probes. The snapshot
always contains every metric key.

This host is a single consumer RTX 5080 (~16 GB). Discovery probes GPU index 0
only. Leave ≥2 GiB free when a later train job or `ci-gpu` needs the card;
this probe is NVML-only and must not start a CUDA context.

## Binding

1. Write one `bkl_gpu_capability` JSON for the run (`agoge_run_id`) **before**
   sampling. The snapshot’s `timestamp_utc` and `monotonic_ns` must not be later
   than bound `bkl_gpu_sample` records.
2. Stamp that snapshot's `capability_digest` onto each `bkl_gpu_sample`.
3. The CPU checker rejects samples whose digest does not match.

Probe **values** are not in the digest, so later samples can change power/util
without invalidating the capability record. Identity, tool versions, and
per-metric support flags are.

## CPU tests

Fake backends (no GPU, no `pynvml`):

| Scenario | What it proves |
|---|---|
| `full-support` | UUID/PCI + versions + all metrics `ok` |
| `partial-support` | `temperature_memory` unsupported; other metrics still sampled |
| `permission-denied` | power/thermals denied (`value` null, not `0`); util continues |
| `transient-failure` | timeout/no-data on some fields; collector continues |
| `device-lost` | sensors `device_lost`; stable UUID still recorded |
| `no-device` | `device_status=no_device`; identity null; metrics unsupported |
| `valid-zero` | `utilization_gpu=0` with `status=ok`; idle power is not `0` |

```bash
python3 tools/check_nvml_capability.py
python3 tools/check_f0_correlation.py
```

`ci-cpu` runs the same commands.

## RTX 5080 evidence

Human-run recipe: [`recipes/f0-nvml-capability.md`](../recipes/f0-nvml-capability.md).
Write live JSON under gitignored `results/`. Do not dual-occupy this 16 GB card
with `ci-gpu` or a train job.

## Validate one file

```bash
python3 tools/check_nvml_capability.py \
  --fixture fixtures/f0-nvml-capability/partial-support.json
python3 tools/check_nvml_capability.py \
  --capability fixtures/f0-correlation/bkl-gpu-capability.json \
  --bind-samples fixtures/f0-correlation/bkl-gpu-samples.jsonl
```
