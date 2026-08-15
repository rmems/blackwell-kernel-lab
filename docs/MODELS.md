# Model cookbook — 16 GB RTX 5080

Guidance for **interactive local agents**. Class rows are **targets**, not guarantees; the measured table below
is what this host actually did. Re-measure after engine updates.

## Happy path

| Class | Examples (illustrative) | Quant | Notes |
|-------|-------------------------|-------|--------|
| Dense 7–9B | Qwen2.5/3-7B/8B, Llama-3.x 8B | Q4_K_M / NVFP4 | Best balance for multi-turn tools — **measured** below |
| Dense 12–14B | Gemma-class ~12B | Q4 / NVFP4 | Watch KV headroom — **measured** below |
| MoE ~20B sparse | GPT-OSS-20B class | MXFP4 / Q4 | Strong when active params stay small — **not measured** (no MoE weights on host) |
| Tiny / draft | 1–3B | Q4–Q8 | Speculative draft or router-side helpers |

## Measured on ShipOfTheseus — 2026-08-15 (#13 / RM-472)

Host: RTX 5080 16303 MiB · driver 610.43.03 · **Ollama 0.30.6** · profile `live_agent_tool_loop` · `max_tokens=96` unless noted.
Idle board before each load: **2061 MiB used / 13817 free** (desktop + browser; no engine).

`resident` is engine-reported (`/api/ps` `size_vram / size`) and now lands in result JSON as `model.residency`.
**VRAM peak** is whole-board `nvidia-smi` (includes the ~2.0 GB desktop); **proc** is the `llama-server` process alone.

| Class | Model (tag) | Quant | ctx | resident | VRAM peak / proc MiB | TTFT cold ms | TTFT resume ms | TPOT p50 ms | Tool-loop p50 ms | Result |
|-------|-------------|-------|-----|----------|----------------------|--------------|----------------|-------------|------------------|--------|
| Dense 7–9B | `granite4.1:8b-ctx8k` (8.8B) | Q4_K_M | 8192 | **100%** | **8748 / 6682** | 2440 (incl. load) | **76.7** | **7.35** | **489** | [json](results/cookbook-2026-08-15/20260815T130312Z-cookbook-granite-ctx8k-tool-loop-f8cdbad8.json) |
| Dense 7–9B, ctx unpinned | `granite4.1:8b` (same weights) | Q4_K_M | 131072 | **47%** | 14918 / 12848 | 6260 | 205 | 45.1 | 2361 | [json](results/cookbook-2026-08-15/20260815T130230Z-cookbook-granite-tool-loop-26deb30b.json) |
| Dense 9B at Q8 | `qwen3.5:9b-q8_0-ctx8k` (9.7B) | Q8_0 | 8192 | **100%** | 12344 / 10272 | 5070 | 1571 | 2.51 ⚠ | 4900 | [json](results/cookbook-2026-08-15/20260815T130410Z-cookbook-qwen9b-q8-ctx8k-512-79ba21f4.json) |
| Dense 12–14B | `phi4:14b-ctx8k` (14.7B) | Q4_K_M | 8192 | **100%** | 12460 / 10390 | 3635 | **80.7** | 11.6 | 3188 | [json](results/cookbook-2026-08-15/20260815T130419Z-cookbook-phi4-14b-ctx8k-a578a712.json) |

### Pin the context or you are not measuring the GPU

Ollama 0.30.6 loads a model at its **advertised** context (131072 for `granite4.1:8b`), which inflates the
allocation to ~27 GB and silently offloads **53% to CPU** on a 16 GB card. Same weights, same prompt, ctx
131072 → 8192: decode **45.1 → 7.35 ms/token (6.1×)**, resume TTFT **205 → 77 ms**, free VRAM **961 → 7131 MiB**.

```bash
printf 'FROM granite4.1:8b\nPARAMETER num_ctx 8192\n' > /tmp/Modelfile.ctx8k
ollama create granite4.1:8b-ctx8k -f /tmp/Modelfile.ctx8k   # reuses existing blobs, no download
ollama ps   # PROCESSOR must read 100% GPU
```

Always check `PROCESSOR` / `model.residency.resident_pct` before quoting any kernel number.

### ⚠ Thinking models break TTFT/TPOT semantics

`qwen3.5:9b-q8_0` (thinking) emits reasoning tokens that count in `usage` but not in the content stream.
TTFT is measured at the first **content** token, so it absorbs the whole reasoning phase (resume 1571 ms) while
TPOT divides a short content tail by all tokens (2.51 ms — **not a real decode rate**). At `max_tokens=96` it
spent the whole budget thinking and returned no content at all
([failed run](results/cookbook-2026-08-15/20260815T130350Z-cookbook-qwen9b-q8-ctx8k-13312af1.json), `slo.met=false`);
the row above is the `--max-tokens 512` retry. For kernel work prefer non-thinking checkpoints, or compare
thinking models only against each other.

### MoE class — not measured

No MoE weights on this host — every local tag reports a dense architecture (`granite`, `qwen35`, `gemma4`,
`phi3`). The MoE ~20B row in the class table stays a **class target**, unmeasured. Deferred rather than
pulling a multi-GB GGUF just to fill a docs row; revisit when an MXFP4 MoE lands on the box for #16 cell D.

### Related measured runs

- L1 kernel ablation (FA on/off, gemma-4 E2B Q4 on llama-server): [docs/results/l1-ablation-2026-08-12/](results/l1-ablation-2026-08-12/) · [MATRIX](MATRIX.md)
- First live multi-phase fixture (granite4.1:8b, unpinned ctx): [docs/results/live-metrics-2026-08-12/](results/live-metrics-2026-08-12/)
- Workload profile definitions: [WORKLOADS.md](WORKLOADS.md)

## Avoid as default interactive agent base

| Class | Why |
|-------|-----|
| Dense 70B full precision | Will not fit; offload kills tool-loop feel |
| Dense 30–70B Q4 with long context | May fit poorly once KV + tools grow |
| “Biggest possible” without cache policy | Multi-agent thrash |
| **Any tag at its advertised max context** | Measured: `granite4.1:8b` at 131k ctx → 47% resident, **6.1× slower decode** |
| Thinking checkpoints for kernel numbers | Reasoning tokens corrupt TTFT/TPOT semantics (see above) |

## Context budgets

| Context | Use |
|---------|-----|
| 4–8k | Default multi-agent / coding tools — 8192 measured 100% resident for 8.8B Q4, 9.7B Q8 and 14.7B Q4 |
| 16k | Single agent RAG-ish tasks |
| 32k+ | Only with measured VRAM; dual-GPU not available on this host |

Ollama ignores these budgets unless the tag pins `num_ctx` — pin it, then confirm with `ollama ps`.

## Agent-quality notes

- Prefer instruction / tool-tuned checkpoints.  
- Constrained JSON / tool schemas matter more than raw MMLU for this lab.  
- Record `model_id`, quant, engine, context, and `vram_peak_mb` in every result JSON.  
- Record residency too: a fast-looking number from a 47%-resident model is measuring the CPU.

## Training

Fine-tunes and training forges: **`rmems/agoge-forger`**, not this lab.
