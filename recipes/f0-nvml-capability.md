# Recipe: F0 NVML capability probe (RTX 5080) — RM-1350

One-shot NVML discovery for the F0 sampler. Records which metrics this
RTX 5080 actually exposes and binds a digest to a run id. Not a sampling
loop ([#53](https://github.com/rmems/blackwell-kernel-lab/issues/53)), not a
daemon, and not a CUDA context.

Contract: [`docs/f0-nvml-capability.md`](../docs/f0-nvml-capability.md).

## CPU path (no GPU)

```bash
python3 tools/check_nvml_capability.py
python3 tools/check_f0_correlation.py
```

Expected: fake-backend scenarios pass, committed fixtures match, correlation
samples carry the capability digest.

## Live probe (ShipOfTheseus)

NVML only. Do **not** start a train job or `ci-gpu` at the same time. Check
`nvidia-smi` first if you will run other GPU work after this; leave **≥2 GiB**
free for those jobs. This probe itself should not allocate VRAM.

```bash
mkdir -p results
python3 tools/check_nvml_capability.py --live \
  --agoge-run-id "local-nvml-cap-$(date -u +%Y%m%dT%H%M%SZ)" \
  --out results/nvml-capability.json \
  --require-device \
  --require-name "NVIDIA GeForce RTX 5080" \
  --require-compute-capability 12.0
python3 tools/check_nvml_capability.py --fixture results/nvml-capability.json
```

Expected:

- `device_status=ok`
- `gpu.name` = `NVIDIA GeForce RTX 5080`
- `gpu.compute_capability` = `12.0`
- `gpu.uuid` and `gpu.pci_bus_id` non-null
- `tools.nvml_version` and `tools.driver_version` non-null
- each metric has `status` in the documented set; missing readings use `value: null`
- `utilization_gpu` may be `0` only when `status` is `ok`

`results/nvml-capability.json` stays gitignored. Commit CPU fixtures only.

If NVML cannot load, the tool still writes a snapshot with
`device_status=unsupported` unless `--require-device` is set (recipe requires
the device).

## Notes

- Arch: **sm_120** / compute **12.0**. Not `sm_100`, FA4, TMEM, or MIG.
- `temperature_memory` is allowed to be `unsupported` on this card.
- Do not substitute `0` for permission errors, timeouts, or a lost device.
