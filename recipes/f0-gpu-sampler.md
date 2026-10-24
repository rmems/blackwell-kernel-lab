# Recipe: F0 GPU sampler around one MiniCPM5 canary — RM-1230

One Agoge QLoRA canary on ShipOfTheseus, with BKL sampling the RTX 5080
for that same `agoge_run_id`. The sampler starts, you run the trainer, the
sampler stops. It does not launch Agoge and it is not a daemon.

Contract: [`docs/f0-gpu-sampler.md`](../docs/f0-gpu-sampler.md).
Markers: Agoge writes `runs/<run_name>/telemetry/agoge-markers.jsonl`
([agoge-forger#147](https://github.com/rmems/agoge-forger/pull/147)).

## CPU path (no GPU)

```bash
python3 tools/check_f0_gpu_sampler.py
python3 tools/check_f0_correlation.py
python3 tools/check_nvml_capability.py
```

Expected: fake-backend scenarios pass, including missing, permission-denied,
and device-lost rows whose `value` is null. This does not accept a live
canary.

## Live canary (ShipOfTheseus)

Do this on the workstation. Pause the self-hosted runner first so
`ci-gpu` / `ci-gpu-sanitizer` cannot share the card. The sampler is NVML
only and does not allocate a CUDA context; the trainer does.

Confirm free VRAM before start. The sampler refuses when free memory is
under **2048 MiB**. The MiniCPM5 canary also needs start-free to cover the
trainer peak plus that floor (about **4597 MiB**); see
[`docs/TRAIN_FIT_5080.md`](../docs/TRAIN_FIT_5080.md). Use the host overlay
that already fits this card. Do not copy trainer YAML into this repo.

```bash
bash kernels/tools/wait_gpu_headroom.sh 2048 1 1
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv
```

If free VRAM is below 2048 MiB, stop. From this repo, with `AGOGE` set to
an agoge-forger checkout:

```bash
set -euo pipefail
RUN_ID="minicpm5-canary-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p results
cleanup() { python3 tools/f0_gpu_sampler.py stop --run-id "$RUN_ID" || true; }
trap cleanup EXIT
python3 tools/f0_gpu_sampler.py start --run-id "$RUN_ID"
(
  cd "$AGOGE"
  uv run agoge train-qlora \
    --config /home/raulmc/agoge-data/scratch/smoke/configs/local_minicpm5_canary.yaml \
    --run-id "$RUN_ID"
)
```

`start` requires compute capability **12.0** unless
`--allow-other-compute-capability` is set. A non-zero `start` skips the
trainer (`set -e`). The `EXIT` trap still calls `stop`, including when
training fails. The sampler process is gone after `stop` or after
`--max-duration-s` (default 7200). Output:

```text
results/f0-sampler/$RUN_ID/capability.json
results/f0-sampler/$RUN_ID/samples.jsonl
results/f0-sampler/$RUN_ID/clock-skew.json
results/f0-sampler/$RUN_ID/overhead.json
```

Join and derive. Point `--markers` at the Agoge run's
`telemetry/agoge-markers.jsonl` (the run directory name is Agoge's, not
`$RUN_ID`):

```bash
python3 tools/check_f0_correlation.py \
  --markers "$AGOGE/runs/<run_name>/telemetry/agoge-markers.jsonl" \
  --samples "results/f0-sampler/$RUN_ID/samples.jsonl" \
  --clock-skew "results/f0-sampler/$RUN_ID/clock-skew.json"
python3 tools/summarize_f0_efficiency.py \
  --markers "$AGOGE/runs/<run_name>/telemetry/agoge-markers.jsonl" \
  --samples "results/f0-sampler/$RUN_ID/samples.jsonl" \
  --out "results/f0-sampler/$RUN_ID/efficiency.json" \
  --report "results/f0-sampler/$RUN_ID/efficiency.md"
```

Expected on a successful canary:

- `device_status=ok`, `gpu.name` = `NVIDIA GeForce RTX 5080`, compute capability `12.0`
- samples carry the same `agoge_run_id` as the markers and a `capability_digest`
- `overhead.json` records `cadence_ms`, jitter, and `collect_duration_ns`
- `floor_held` is true only when sampled free memory stayed ≥ 2048 MiB
- missing sensors stay null; they do not show up as `0`

`results/` stays gitignored. CPU fixtures are the CI evidence. A live
overhead number and a joined MiniCPM5 bundle are produced by running this
recipe on ShipOfTheseus; `ci-cpu` does not do that.

## Notes

- Arch: **sm_120** / compute **12.0**. Not `sm_100`, FA4, TMEM, or MIG.
- Default sample interval is 1000 ms. The overhead budget is a median
  collect time of at most 50 ms and at most 5% of that interval.
- `temperature_memory` may be `unsupported` on this card.
- `profile_window_ref` is left null. Agoge's optional `train:1-1` window
  is a separate artifact; this sampler does not guess which samples fall
  inside it.
