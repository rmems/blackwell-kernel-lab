# Recipe: L2 Green Context contention (sm_120) — #17 / RM-471

Measures whether two CUDA Green Context SM partitions protect a short,
latency-sensitive first-party kernel from a concurrent multi-wave background
kernel on this RTX 5080. This is an L2 host-scheduling benchmark: it is not MIG,
a model-serving benchmark, or evidence that an inference engine automatically
inherits the same improvement.

## Build and run

```bash
cmake -S kernels -B build/kernels -DBKL_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DBKL_CUDA_HOST_COMPILER=/home/linuxbrew/.linuxbrew/bin/g++-15
cmake --build build/kernels -j"$(nproc)"
./build/kernels/src/bkl_green_ctx_bench \
  --out results/green-ctx-bench.json
python3 -m json.tool results/green-ctx-bench.json >/dev/null
```

The executable queries the device resource rather than assuming this card has
84 SMs. It derives a legal symmetric split from `minSmPartitionSize` and
`smCoscheduledAlignment`, generates two resource descriptors, creates two
Green Contexts, and destroys their streams before their contexts.

Expected terminal banner:

```text
bkl green_ctx_bench sm_120
outcome=measured sm_total=… min_partition=… alignment=…
split sensitive=… background=… remainder=…
invocation=1 … gate=pass|fail
invocation=2 … gate=pass|fail
invocation=3 … gate=pass|fail
follow_up_justified=true|false
bkl green_ctx_bench sm_120 ok
```

The final `ok` sentinel means the benchmark produced a structurally valid
result. It does not mean the 10% decision gate passed. Read `outcome` and
`decision.follow_up_justified` from the JSON.

## Measurement design

- The ordinary baseline uses two non-blocking streams and gives the sensitive
  stream the device's highest available priority; the background stream uses
  the lowest priority. The comparison therefore does not handicap the
  ordinary-stream baseline by ignoring stream priorities.
- The background kernel launches three waves of 1,024-thread blocks. The grid
  is derived from CUDA's reported active-block occupancy and total SM count.
  Each block runs for about 10 ms, and a mapped readiness counter ensures the
  first resident wave is executing before the sensitive kernel is submitted.
- The partitioned path runs the identical background and sensitive kernels on
  two disjoint Green Context streams with the same priorities.
- CUDA events on the sensitive stream record submission-to-completion GPU
  latency, including the scheduling delay behind already-running background
  blocks. Host launch-to-synchronization time is retained separately.
- Each mode receives one unrecorded complete warmup and seven retained samples.
  Three independent in-process invocations recreate the Green Contexts and
  alternate order: ordinary→partitioned, partitioned→ordinary, then
  ordinary→partitioned.
- Every sensitive output element and every background block checksum is
  validated outside the timing window.

Before allocation, the executable records `cudaMemGetInfo`, its explicit
planned allocation, and projected remaining VRAM. It also samples free VRAM
after allocation and around both execution modes. A projected or observed value
below 2 GiB produces an honest `insufficient_vram_headroom` outcome with no
latency distributions and explicitly null summary latency fields.

## Outcome model

Exit status 0 plus the final sentinel is limited to these report outcomes:

| `outcome` | Meaning |
|---|---|
| `compile_time_api_unavailable` | Installed CUDA headers predate the Driver API surface. |
| `unsupported` | The Driver API returned `CUDA_ERROR_NOT_SUPPORTED`. |
| `not_permitted` | The Driver API returned `CUDA_ERROR_NOT_PERMITTED`. |
| `no_legal_split` | Queried resource constraints cannot form two valid partitions. |
| `insufficient_vram_headroom` | Planned or observed free VRAM would fall below 2 GiB. |
| `measured` | Both paths completed with valid output and full distributions. |

Unexpected initialization, query, allocation, launch, synchronization,
correctness, or JSON-publication failures exit nonzero without the sentinel.
The JSON is first written to a same-directory temporary file and then renamed
into place, so a partial result is never published as complete.

## Measured result — ShipOfTheseus, 2026-08-30

Environment: RTX 5080 (`sm_120`), driver 610.43.03, CUDA runtime/headers 13.3.
The runtime query returned:

| Resource | Value |
|---|---:|
| Total SMs | 84 |
| Minimum partition | 8 SMs |
| Co-scheduled alignment | 8 SMs |
| Sensitive partition | 40 SMs |
| Background partition | 40 SMs |
| Remainder | 4 SMs |

The retained medians were:

| Invocation | Order | Ordinary | Partitioned | Reduction |
|---:|---|---:|---:|---:|
| 1 | ordinary→partitioned | 9.5092 ms | 0.1407 ms | 98.52% |
| 2 | partitioned→ordinary | 9.5218 ms | 0.1408 ms | 98.52% |
| 3 | ordinary→partitioned | 9.5248 ms | 0.1410 ms | 98.52% |

All three invocations passed the required 10% reduction independently and all
output checks passed. Explicit planned device allocation was 6,112 bytes; the
minimum observed free VRAM was 14,052,818,944 bytes, comfortably above the
2 GiB rule.

This result justifies a separate engine-integration experiment. In particular,
`agoge-forger` is expected to use vLLM, so a downstream experiment should test
vLLM worker/process ownership, CUDA Graph capture, stream creation, allocator
behavior, and KV-cache residency explicitly. The microbenchmark does not prove
that vLLM can adopt Green Contexts unchanged, and vLLM orchestration does not
belong in this repository.

Hard boundary: **sm_120 ≠ sm_100**. Make no FA4, TMEM, B200, or MIG inference
from this result.
