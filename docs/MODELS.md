# Model cookbook — 16 GB RTX 5080

Guidance for **interactive local agents**. Numbers are **class targets**, not guarantees; re-measure after engine updates.

## Happy path

| Class | Examples (illustrative) | Quant | Notes |
|-------|-------------------------|-------|--------|
| Dense 7–9B | Qwen2.5/3-7B/8B, Llama-3.x 8B | Q4_K_M / NVFP4 | Best balance for multi-turn tools |
| Dense 12–14B | Gemma-class ~12B | Q4 / NVFP4 | Watch KV headroom |
| MoE ~20B sparse | GPT-OSS-20B class | MXFP4 / Q4 | Strong when active params stay small |
| Tiny / draft | 1–3B | Q4–Q8 | Speculative draft or router-side helpers |

## Avoid as default interactive agent base

| Class | Why |
|-------|-----|
| Dense 70B full precision | Will not fit; offload kills tool-loop feel |
| Dense 30–70B Q4 with long context | May fit poorly once KV + tools grow |
| “Biggest possible” without cache policy | Multi-agent thrash |

## Context budgets

| Context | Use |
|---------|-----|
| 4–8k | Default multi-agent / coding tools |
| 16k | Single agent RAG-ish tasks |
| 32k+ | Only with measured VRAM; dual-GPU not available on this host |

## Agent-quality notes

- Prefer instruction / tool-tuned checkpoints.  
- Constrained JSON / tool schemas matter more than raw MMLU for this lab.  
- Record `model_id`, quant, engine, context, and `vram_peak_mb` in every result JSON.

## Training

Fine-tunes and training forges: **`rmems/agoge-forger`**, not this lab.
