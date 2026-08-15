# Recipe: kernel ablation (#16 / RM-470)

**L1 kernel campaign:** measure Flash Attention × CUDA graphs × quant on **sm_120**.

Requires [engine-smoke.md](engine-smoke.md) green.

## What we are measuring

| Lever | Where it lives | Agent effect |
|-------|----------------|--------------|
| Quant (Q4 / MXFP4 / NVFP4) | Model weights / engine | Fit + decode bandwidth |
| Flash Attention | Engine CUDA kernels | Prefill TTFT, KV memory |
| CUDA graphs | Engine CUDA runtime | Short tool decode TPOT |

**Not:** writing new `.cu` yet. **Not:** FA4/B200/TMEM recipes. **sm_120 ≠ sm_100** (no MIG either).

## Cells

| Cell | Intent |
|------|--------|
| A | Baseline engine defaults |
| B | FlashAttn forced on (if flag exists) |
| C | Graphs on (if flag exists) |
| D | Alternate quant (MXFP4/NVFP4) if available |

### Ollama path (flags limited)

Ollama often hides low-level FA/graph toggles. Use it for smoke (#12) or quant D-style comparisons; use **llama-server** for FA cells.

### llama-server path (ShipOfTheseus — measured)

```bash
# Homebrew llama.cpp 9190
MODEL=/path/to/gemma-4-E2B-it-UD-Q4_K_XL.gguf bash harness/serve/run_l1_ablation.sh
```

| Cell | Flags | Status |
|------|-------|--------|
| A | `-fa off -ngl 99 -c 4096` | Measured |
| B | `-fa on` | Measured |
| C | `-fa on` + CUDA graphs **engine-default** | Measured — **no CLI graph toggle** on 9190 |
| D | MXFP4/NVFP4 twin | Deferred (no twin GGUF on host) |

Snapshots: `docs/results/l1-ablation-2026-08-12/`. MATRIX rows: `docs/MATRIX.md`.

Manual one-off:

```bash
llama-server -m "$MODEL" -ngl 99 -c 4096 --port 8080 -fa off   # or on
export BKL_BASE_URL=http://127.0.0.1:8080/v1
export BKL_MODEL=gemma-4-E2B-it-UD-Q4_K_XL.gguf
python3 harness/serve/smoke_openai.py --out results/ --stream --skip-content-slo \
  --agent-prompt --label cell-A-fa-off --engine-flags "-ngl 99 -c 4096 -fa off" \
  --engine-version "llama-server 9190" --quant UD-Q4_K_XL
```

## Prefill-heavier cells (#14 profiles)

`--agent-prompt` is a ~40-token prompt — too short for a strong FA signal. Reuse a named workload profile's
cold-prefill turn instead, so the ablation cell and the live agent row measure the same bytes
([WORKLOADS.md](../docs/WORKLOADS.md)):

```bash
python3 harness/serve/smoke_openai.py --out results/ --stream --skip-content-slo \
  --profile live_coding_tool --label cell-B-fa-on-coding \
  --engine-flags "-ngl 99 -c 4096 -fa on" --engine-version "llama-server 9190"
```

Recorded as `workload.profile = live_coding_tool_cold_only` (single turn, no resume phase). For the full
cold→tool→resume loop use `run_live_metrics.py --profile …` against the same server.

## Success for #16

- [x] ≥3 cells with TTFT + wall + VRAM (A–C)  
- [x] MATRIX row(s) updated  
- [x] AGENT_STACK notes flags on this host  
- [x] Explicit fallback: graphs not independently toggleable; D deferred  
