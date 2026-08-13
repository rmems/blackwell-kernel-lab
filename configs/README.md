# Engine configs

| File | Purpose |
|------|---------|
| `engine.example.env` | Template for OpenAI-compatible client (#3) |
| `engine.local.env` | Local overrides (**gitignored** — do not commit secrets) |

```bash
cp configs/engine.example.env configs/engine.local.env
# edit model / base URL, then:
set -a && source configs/engine.local.env && set +a
python3 harness/serve/smoke_openai.py --stream
python3 harness/agent_loop/run_live_metrics.py --out results/
```
