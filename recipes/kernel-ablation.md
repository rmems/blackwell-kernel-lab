# Recipe: kernel ablation (#16 / RM-470)

**L1 kernel campaign:** measure Flash Attention × CUDA graphs × quant on **sm_120**.

Requires [engine-smoke.md](engine-smoke.md) green.

## What we are measuring

| Lever | Where it lives | Agent effect |
|-------|----------------|--------------|
| Quant (Q4 / MXFP4 / NVFP4) | Model weights / engine | Fit + decode bandwidth |
| Flash Attention | Engine CUDA kernels | Prefill TTFT, KV memory |
| CUDA graphs | Engine CUDA runtime | Short tool decode TPOT |

**Not:** writing new `.cu` yet. **Not:** FA4/B200 recipes.

## Cells

| Cell | Intent |
|------|--------|
| A | Baseline engine defaults |
| B | FlashAttn forced on (if flag exists) |
| C | Graphs on (if flag exists) |
| D | Alternate quant (MXFP4/NVFP4) if available |

### Ollama path (flags limited)

Ollama often hides low-level FA/graph toggles. For Ollama:

1. Run baseline smoke + timed completion → cell **A**  
2. Compare **two models/quants** of similar size → cell **D-style** quant comparison  
3. For full FA/graph ablations, use **llama-server** with documented flags:

```bash
# Example shape only — adjust -m path and flag names for your llama.cpp build
llama-server -m /path/model-Q4_K_M.gguf -ngl 99 -c 8192 --port 8080
# Rebuild/run with flash-attn + graph options per llama.cpp docs for your version
```

Record exact binary version and flags in `results/*.json` `notes`.

## Run timed cells via harness

```bash
export BKL_BASE_URL=http://127.0.0.1:11434/v1
export BKL_MODEL=granite4.1:8b

python3 harness/serve/smoke_openai.py --out results/ --label cell-A-baseline --stream
# change engine flags / model, then:
python3 harness/serve/smoke_openai.py --out results/ --label cell-D-alt-quant --stream

python3 harness/report/aggregate.py --results results/
```

## Success for #16 (partial → full)

- [ ] ≥2 cells with TTFT + wall + VRAM notes  
- [ ] MATRIX row(s) updated  
- [ ] AGENT_STACK notes which flags applied on this host  
