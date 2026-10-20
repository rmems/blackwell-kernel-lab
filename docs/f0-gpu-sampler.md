# F0 per-run GPU sampler

**Issue:** [#53](https://github.com/rmems/blackwell-kernel-lab/issues/53) ·
Linear [RM-1230](https://linear.app/rpd-34/issue/RM-1230).

**Audience:** a person running one Agoge training command on ShipOfTheseus.
The sampler records device state for that `agoge_run_id` and then exits.

This is the collection loop that capability discovery
([`f0-nvml-capability.md`](f0-nvml-capability.md)), clock calibration
([`f0-correlation-schema.md`](f0-correlation-schema.md)), and efficiency
derivation ([`f0-efficiency-metrics.md`](f0-efficiency-metrics.md)) were
built for. It is not a daemon, not a Prometheus service, and not the
trainer. Agoge stays the owner of the process model, markers, and phase.
The human-run MiniCPM5 steps are
[`recipes/f0-gpu-sampler.md`](../recipes/f0-gpu-sampler.md).

## What one run writes

Under `results/f0-sampler/<agoge_run_id>/` (gitignored `results/`):

| File | Kind |
|---|---|
| `capability.json` | `bkl_gpu_capability` from `tools/nvml_capability.py`, written before samples |
| `samples.jsonl` | `bkl_gpu_sample` (`bkl.f0_correlation.v1`) |
| `clock-skew.json` | `bkl_clock_skew` from `tools/f0_clock.py` start and end observations |
| `overhead.json` | `bkl.f0_sampler_overhead.v1` sidecar (not a join record) |
| `session.json` | pid, cadence, and paths for this run |
| `refusal.json` | written when start refuses; no sample rows are invented |

`stop` writes the skew and overhead files and clears the pid. A second
`stop` is safe. `--max-duration-s` (default **7200**) ends the process if
`stop` never arrives. Nothing is installed as a service.

## CLI

```bash
python3 tools/f0_gpu_sampler.py start --run-id <id>
python3 tools/f0_gpu_sampler.py stop --run-id <id>
python3 tools/f0_gpu_sampler.py run --run-id <id> -- <command...>
```

`run` only brackets the argv you pass: it samples, waits for that process,
then stops, including when the command exits non-zero. It does not parse
Agoge configs or own the trainer. The recipe uses `start` / `stop`.

The trainer stays in `run`'s process group. There is no `setpgid` and no
terminal handover. While `run` waits it ignores SIGINT, so a terminal
Ctrl+C reaches the trainer once. It forwards only SIGTERM, to that child.
Before exec, the child restores SIGPIPE, SIGTTIN, SIGTTOU, SIGXFSZ, SIGINT,
and SIGTERM to `SIG_DFL`. A stopped child is killed so the bracket can
still write `clock-skew.json` and `overhead.json`.

Samples copy `capability_digest`, host, and GPU identity from the snapshot.
`profile_window_ref` stays null. Phase, step, and token counters are not
inferred from utilization. Join them from Agoge's
`runs/<run_name>/telemetry/agoge-markers.jsonl`
([agoge-forger#147](https://github.com/rmems/agoge-forger/pull/147)).

## Measurements

NVML only (`tools/nvml_live.py` on the host, `tools/nvml_fakes.py` in CPU
CI). nvidia-smi is not a metric source. No CUDA context is created.

Each physical field is a `{value, unit, status}` object. `unavailable`,
`unsupported`, `permission_denied`, `transient_failure`, and `device_lost`
store `value: null`. A real zero stays `status: ok`. `performance_state`
is an additive measurement (`unit: pstate`) so a later consumer can see
P-state without a schema bump. `cuda.runtime_version` stays null; the
sampler does not initialize the CUDA runtime to read it.
`cuda.tool` is `nvml`.

Planned interval is `cadence_ms` (default **1000**). After the first
sample, `observed_interval_ns` and `jitter_ns` record the actual gap.
`collect_duration_ns` is how long that reading took.

## Overhead budget

The collector must stay small relative to a training step. `overhead.json`
sets `overhead_ok` when the median `collect_duration_ns` is at most
**50 ms** and at most **5%** of the planned cadence. At the default
1000 ms cadence both caps are 50 ms. A run with no samples leaves
`overhead_ratio` and `overhead_ok` null. CPU fakes measure this accounting;
they are not the ShipOfTheseus overhead number.

## Headroom

The host floor is **2048 MiB** free (`docs/HOST_BASELINE.md`). `start`
refuses (exit 3, `refusal.json`, no samples) when `headroom` / `vram_free`
is below that floor or the reading is not `ok`. Exit 2 means there is no
GPU identity or the compute capability is not **12.0**.
`--allow-low-headroom` records a breach instead of refusing; the overhead
sidecar then has `floor_held: false`. The sampler does not stop the
trainer and is not an exclusive lock. Train and `ci-gpu` still serialize.

Consumer Blackwell **sm_120** / compute **12.0** only. No MIG, FA4, or TMEM.

## CPU validation

```bash
python3 tools/check_f0_gpu_sampler.py
```

`ci-cpu` runs that command. It drives the NVML fakes through full support,
partial support, a missing (`unavailable`) reading, permission denied,
device lost, a real zero, no device, a headroom refusal, jitter, the
duration bound, and process start/stop. Sample rows are checked with
`tools/check_f0_correlation.py` and bound with the capability digest.
Live NVML on ShipOfTheseus is the recipe, not this test.
