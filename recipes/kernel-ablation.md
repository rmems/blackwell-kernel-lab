# Recipe: kernel ablation (#16 / RM-470)

**L1 kernel campaign:** measure engine CUDA flags and paths — Flash Attention,
CUDA graphs, and quantization — on **sm_120**.

## What L1 measures

| Lever | Engine CUDA primitive | Measurement focus |
|---|---|---|
| Quant (Q4 / MXFP4 / NVFP4) | Quantized weight/GEMM path | Fit and decode-bandwidth tradeoff |
| Flash Attention | Fused attention kernels | Prefill and KV-memory behavior |
| CUDA graphs | CUDA runtime capture/replay | Launch and short-decode behavior |

**Not:** writing new `.cu` yet. **Not:** FA4/B200/TMEM recipes. **sm_120 ≠
sm_100**; do not assume MIG or datacenter-only features.

## Cells

| Cell | Intent |
|---|---|
| A | Baseline engine defaults |
| B | FlashAttn forced on, if the engine exposes a flag |
| C | CUDA graphs on or engine-default, if independently configurable |
| D | Alternate quant (MXFP4/NVFP4), if available |
