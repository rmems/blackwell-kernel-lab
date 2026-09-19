# Train-fit notes — RTX 5080 (~16 GB / sm_120)

Host-grounded QLoRA fit for [`agoge-forger`](https://github.com/rmems/agoge-forger)
on **ShipOfTheseus**. This is the forge-facing ceiling, not an agent-workload
cookbook (closed [#13](https://github.com/rmems/blackwell-kernel-lab/issues/13)
stays not-planned).

**Audience:** agoge-forger. Trainer YAML lives in the forge; this lab records
whether those knobs fit the card.

## Shared ceiling

From [`HOST_BASELINE.md`](HOST_BASELINE.md):

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5080 |
| VRAM | **16303 MiB** (~16 GB) |
| Compute | **12.0** (`sm_120`) — not `sm_100` |

Leave **≥2 GiB (2048 MiB) free** for the desktop and CUDA context. Train and
kernel CI must not both assume the full card. Occupancy is a **snapshot
floor** (`nvidia-smi` / `wait_gpu_headroom.sh`), not an exclusive GPU lock —
`ci-gpu` can still start between a headroom check and allocation. Pause the
runner or wait; see [`CI.md`](CI.md) and the README checklist. Fail closed if
free VRAM is `< 2048 MiB` **before** a train job.

Rough budget:

```text
[ QLoRA / kernel allocation ][ CUDA context ][ OS / display ][ ≥2 GiB headroom ]
```

## Normative train knobs (forge-owned)

Do not copy trainer YAML into this repo. Cite these Agoge configs:

| Path | Role |
|---|---|
| [`configs/minicpm5_canary.yaml`](https://github.com/rmems/agoge-forger/blob/main/configs/minicpm5_canary.yaml) | Current canary knobs on AF `main` (4-bit NF4, seq 512, batch 1, accum 8, grad checkpoint). Not a pin of the 2026-09-11 row. |
| [`configs/granite_4_1_flagship.yaml`](https://github.com/rmems/agoge-forger/blob/main/configs/granite_4_1_flagship.yaml) | Flagship QLoRA **template** (seq 2048, same 4-bit shape) until AF [#101](https://github.com/rmems/agoge-forger/issues/101) revision freeze is measured here |

Preflight warnings (`estimate_training_risk`, ≤16.5 GiB gates) stay in the
forge. This document does not change those thresholds.

## Measurement protocol (MiniCPM5 canary)

Run on ShipOfTheseus only. Capture a one-object JSON under gitignored
`results/` (for example `results/train-fit-minicpm5-canary.json`). Paste the
peak-VRAM / min-free row into the table below.

1. Confirm GPU 0 has **≥2048 MiB free** and **no unexpected compute
   process** (README checklist). Pause the self-hosted runner before train.
   `wait_gpu_headroom.sh` is a snapshot, not a mutex:

   ```bash
   bash kernels/tools/wait_gpu_headroom.sh 2048 1 1
   nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv
   nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv
   ```

   If free VRAM is below 2048 MiB, **stop**. To keep **≥2048 MiB free while
   the trainer is allocated**, start-free must cover that floor **plus** the
   expected peak (≈2549 + 2048 ≈ **4597 MiB** for the MiniCPM5 row below).
   A start-free between 2048 MiB and peak+2048 can still OOM the reserve;
   do not treat that as headroom-compliant unless min-free during the run
   stays ≥2048 MiB.

2. From an `agoge-forger` checkout, use a config that satisfies current train
   provenance (Agoge **#148+**): content-addressed Hub `revision` and a
   `dataset_path` under a **frozen** split (`split_manifest.json` beside the
   split tree). Stock [`configs/minicpm5_canary.yaml`](https://github.com/rmems/agoge-forger/blob/main/configs/minicpm5_canary.yaml)
   still points at `datasets/samples/tiny_sft.jsonl` with no `revision` — it
   will fail with `cannot construct producer_provenance` until AF updates that
   file. The 2026-09-11 row used a host overlay with pinned revision and
   `~/agoge-data/splits/tiny-sft-smoke-v1` (not a pin of `main` yaml).

   ShipOfTheseus replay (matches the proven row):

   ```bash
   uv run agoge train-qlora --config /home/raulmc/agoge-data/scratch/smoke/configs/local_minicpm5_canary.yaml
   ```

   Or an equivalent overlay: same knobs as the canary, plus `revision:
   "156170697656c48f69915b33a2fb44110242187c"` and `dataset_path` set to the
   frozen train shard (for example
   `.../tiny-sft-smoke-v1/splits/train.jsonl`).

3. Record **peak allocated** (trainer / `torch.cuda.max_memory_allocated`) and
   **minimum free** (`nvidia-smi` `memory.free` while the process is alive,
   not after teardown). Snapshot `nvidia-smi` again after the process exits.

Do not infer training phase from GPU utilization. Phase identity is Agoge’s
(`agoge_marker.phase` once [#144](https://github.com/rmems/agoge-forger/issues/144)
emits markers).

### Correlated bundle (do not invent a second format)

[#52](https://github.com/rmems/blackwell-kernel-lab/issues/52) / RM-1229 defined
[`f0-correlation-schema.md`](f0-correlation-schema.md)
(`bkl.f0_correlation.v1`). When [#53](https://github.com/rmems/blackwell-kernel-lab/issues/53)
(sampler), Agoge [#144](https://github.com/rmems/agoge-forger/issues/144)
(markers), and [#55](https://github.com/rmems/blackwell-kernel-lab/issues/55)
(derived peak VRAM / min headroom) are ready:

- Stamp a stable Agoge `agoge_run_id` on the canary.
- Link the BKL JSONL bundle + #55 summary from the table’s **Bundle** column.
- Keep this table as the concise train-fit summary. Peak VRAM / min headroom
  here must agree with the #55 derivation within that issue’s aggregation
  semantics.

Until then the dated row below is the summary; there is no parallel telemetry
schema in this document.

## MiniCPM5 canary — measured

| Date (UTC) | Config | Peak allocated | Start free | Min free during run | Result | Bundle |
|---|---|---|---|---|---|---|
| 2026-09-11 | AF MiniCPM5 QLoRA canary (4-bit NF4, seq 512, bs 1, accum 8, grad ckpt; Hub rev `156170697656c48f69915b33a2fb44110242187c`; frozen split `tiny-sft-smoke-v1`; Agoge commit / yaml digest **not recorded**) | **2.49 GiB** (~2549 MiB) trainer max | not captured (`nvidia-smi` protocol) | not captured | no OOM; trainer peak only | pending `agoge_run_id` + #53/#55 |
| 2026-09-19 | Host overlay `local_minicpm5_canary.yaml` (same 4-bit / seq 512 / bs 1 / grad ckpt; **accum 4**; same Hub rev + frozen split; Agoge `76c3fa3f`) | **2.49 GiB** (~2550 MiB) trainer max | **13503 MiB** | **10358 MiB** (`nvidia-smi` free, step loop) | no OOM; 18 steps, 10.6 s, `train_loss` 2.683; min-free ≥2048 MiB headroom floor | pending `agoge_run_id` + #53/#55; local JSON `results/train-fit-minicpm5-canary.json` |

**Provenance:** The 2026-09-11 row is Agoge Trainer lane on ShipOfTheseus, recorded on
[`agoge-forger#104`](https://github.com/rmems/agoge-forger/issues/104#issuecomment-5629835314)
(mechanical qualify; 18 steps, 8.6 s). Peak is trainer-reported max VRAM, not
an `nvidia-smi` used-memory sample. It shows the canary did not OOM at
**2.49 GiB allocated** on this 16 GB card; it does **not** validate start-free
/ min-free against the ≥2 GiB snapshot floor (context and other processes
were not sampled). It is also **not** a correlated #51–#55 bundle.

The 2026-09-19 row replays the protocol on this host (preflight
`wait_gpu_headroom.sh`, start/min-free `nvidia-smi` sampling during
`uv run agoge train-qlora` with the pinned-revision overlay). Start-free
**13503 MiB** exceeds peak+2048 (~4597 MiB implied used at peak); min-free
**10358 MiB** stayed above the 2048 MiB floor with desktop GPU apps present.
Still not a correlated #51–#55 bundle until `agoge_run_id` lands.

Do not treat the `main` yaml URL as the pin for either row.

## Granite 4.1 — template until measured

**Unmeasured** against this protocol and against
[`configs/granite_4_1_flagship.yaml`](https://github.com/rmems/agoge-forger/blob/main/configs/granite_4_1_flagship.yaml)
(seq **2048**).

AF [#101](https://github.com/rmems/agoge-forger/issues/101) still owns the
revision freeze and the first measured G0/G1 experiment. Repeat the MiniCPM5
protocol with that flagship config **after** the freeze lands. Do not treat
mechanical-qualify VRAM from [#104](https://github.com/rmems/agoge-forger/issues/104)
(seq 512 tiny / cap-512 R2EGym) as a seq-2048 flagship train-fit row.

| Date (UTC) | Config | Peak allocated | Start free | Min free during run | Result | Bundle |
|---|---|---|---|---|---|---|
| — | `granite_4_1_flagship.yaml` (seq 2048, 4-bit NF4, bs 1, accum 8, grad ckpt) | **unmeasured** | — | — | template | — |

## Serve / contention context (not train knobs)

L1 prefix/KV and L2 Green Context numbers describe **inference/serve
contention** on this card. They are not QLoRA hyperparameters.

- L1 prefix/KV reuse: [`recipes/l1-prefix-kv-reuse.md`](../recipes/l1-prefix-kv-reuse.md)
  — measured prompt-eval speedup with 3.4–3.5 GiB still free (2026-08-30).
- L2 Green Context isolation: [`recipes/l2-green-ctx-bench.md`](../recipes/l2-green-ctx-bench.md)
  — SM partitioning for a latency-sensitive kernel; not the default for
  train vs Actions. Train ↔ `ci-gpu` still share the 2 GiB snapshot floor
  (not a mutex).

## Non-goals

- Multi-GPU / cloud (Dioscuri)
- Expanding AF preflight thresholds from this page
- Agent-loop or product workload profiles
- A second telemetry format beside `bkl.f0_correlation.v1`
