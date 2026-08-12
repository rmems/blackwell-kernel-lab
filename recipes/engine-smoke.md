# Recipe: engine smoke (#12 / RM-476) — unlocks kernel work

**Goal:** Primary engine serves a small model on RTX 5080 so **L1 kernels** (FA, graphs, quant) can be exercised.

## Primary engine (this host)

**Ollama** (`0.30.6+`) — OpenAI-compatible at `http://127.0.0.1:11434/v1`.

Alternate: `llama-server` (Homebrew) once a GGUF path is chosen.

## Smoke

```bash
cd ~/rmems/blackwell-kernel-lab

# Optional: source configs/engine.local.env
export BKL_BASE_URL=http://127.0.0.1:11434/v1
export BKL_MODEL=granite4.1:8b   # ~5–10GB class; leave headroom

python3 harness/serve/smoke_openai.py --out results/
```

Or raw curl:

```bash
curl -sS "$BKL_BASE_URL/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$BKL_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: kernel-smoke-ok\"}],\"max_tokens\":16,\"temperature\":0}"
```

## After smoke

```bash
ollama stop "$BKL_MODEL"   # free VRAM for desktop / next run
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
```

**Headroom rule:** prefer ≥2 GB free after load when multi-tasking; Ollama may hold the model resident until `stop`.

## Next (kernel campaign)

→ [kernel-ablation.md](kernel-ablation.md) (#16 / RM-470)
