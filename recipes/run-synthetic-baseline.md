# Recipe: synthetic agent baseline

No live model required. Validates harness + GPU probe.

```bash
# From the repository root
python3 harness/agent_loop/run_synthetic.py --out results/ --profile synthetic_react --steps 5
python3 harness/report/aggregate.py --results results/
```

Expect a `results/*.json` file and printed p50 metrics. GPU name/driver filled when `nvidia-smi` works.
