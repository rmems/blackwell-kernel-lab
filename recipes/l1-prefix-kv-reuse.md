# L1 prefix / KV reuse on sm_120

This recipe measures an inference engine's observable prefix-cache behavior on
the RTX 5080. It compares the same model, context size, static text, and varying
task in two layouts:

| Cell | Prompt layout | Expected reuse |
|---|---|---|
| `prefix_first` | `[stable policy + tool schemas][varying task]` | The long common prefix can be reused. |
| `varying_first` | `[varying task][stable policy + tool schemas]` | The first changed token invalidates almost all prefix reuse. |

This is an L1 engine measurement, not an agent-memory product, a session-store
benchmark, or evidence that a custom KV CUDA kernel is needed.

## Policy

- **Keep** system policy, tool schemas, and other byte-stable context at the
  start of a prompt.
- **Drop** stale or request-specific material instead of preserving it merely
  to lengthen a reusable prefix.
- **Reuse** only when engine version, model, context size, sampling options,
  and static policy are unchanged. Put the varying user task after that prefix.
- Invalidate the comparison after any of those inputs change; cache reuse is an
  optimization, never a correctness dependency.

## Prerequisites

- Ollama is already running on `127.0.0.1:11434`.
- The selected model already exists locally and fits entirely on the GPU. The
  recipe deliberately calls `ollama show`, not `ollama pull`.
- `curl`, `flock`, `jq`, `ln`, `nvidia-smi`, `ollama`, `rg`, and `sha256sum`
  are available.
- At least **2 GiB of VRAM remains free while the model is loaded**.
- The expected full prompt-token count is known. The default `1466` is pinned
  to the default model and 96-line prefix so silent context truncation fails
  closed. If either changes, first use a context known to fit comfortably,
  observe the stable full `prompt_eval_count`, then set
  `BKL_EXPECTED_PROMPT_TOKENS` to that reviewed value.

The default below uses the already-local model measured on ShipOfTheseus. Set
`BKL_MODEL` to another already-local model when reproducing elsewhere.

## Run

Run [`scripts/l1-prefix-kv-reuse.sh`](scripts/l1-prefix-kv-reuse.sh) from the
repository root:

```bash
./recipes/scripts/l1-prefix-kv-reuse.sh
```

Every setting is an environment variable with the ShipOfTheseus default baked
in, so reproducing elsewhere means overriding the ones that differ:

```bash
BKL_MODEL=qwen3:8b \
BKL_EXPECTED_PROMPT_TOKENS=1502 \
BKL_RESULT_NAME=prefix-kv-reuse-qwen3.jsonl \
  ./recipes/scripts/l1-prefix-kv-reuse.sh
```

| Variable | Default | Meaning |
|---|---|---|
| `BKL_MODEL` | `phi4:14b` | Already-local model. The script calls `ollama show`, never `ollama pull`. |
| `BKL_BASE_URL` | `http://127.0.0.1:11434` | Must be a canonical IPv4-loopback origin. |
| `BKL_NUM_CTX` | `8192` | Context size; must fit the expected prompt plus one token. |
| `BKL_PREFIX_LINES` | `96` | Lines of byte-stable policy in the reusable prefix. |
| `BKL_EXPECTED_PROMPT_TOKENS` | `1466` | Pinned so silent context truncation fails closed. |
| `BKL_RUNS` | `3` | Independent invocations; minimum 3. |
| `BKL_MIN_FREE_MIB` | `2048` | The 2 GiB host headroom rule; cannot be lowered. |
| `BKL_CONNECT_TIMEOUT` | `5` | Seconds, per local API call. |
| `BKL_REQUEST_TIMEOUT` | `120` | Seconds, whole request. |
| `BKL_RESULT_NAME` | `prefix-kv-reuse.jsonl` | Basename only; output always lands under `results/`. |

The script exits nonzero unless the decision gate passes, and prints the
summary JSON to stdout.

## What the script enforces

- **Refuses to overwrite** an existing result, and takes an endpoint-keyed
  `flock` so two runs cannot interleave against the same Ollama instance.
- **Stops the model between cells** so every prime starts cold, and stops it
  again on exit. The prime request is excluded from the measured request.
- **Alternates cell order** across invocations (prefix→varying, varying→prefix,
  prefix→varying) so ordering cannot explain the result.
- **Exercises each cache path once before recording.** On this host the
  engine's first-ever prefix-reuse request can include one-time initialization
  even after its prime request has returned.
- **Rejects a prompt-token count** that differs from `BKL_EXPECTED_PROMPT_TOKENS`,
  so silent context truncation or tokenizer drift fails closed rather than
  producing a fast, wrong number.
- **Requires the model to be 100% GPU-resident** and at least
  `BKL_MIN_FREE_MIB` free before and during every measured request.
- **Requires exactly one streamed response token** with a non-empty chunk, and
  resolves an immutable model digest from `/api/tags` into every record.
- **Records streaming TTFT** from `curl`'s `time_starttransfer` on the streaming
  response, alongside `time_total` through one generated token.
- **Publishes atomically**: records accumulate in a scratch directory and are
  hard-linked into place, so a concurrently created result is never clobbered.

Both the recorded pass and the warmup pass build their prompts through one
`build_cell_prompts` helper, so the warmup cannot drift from the layout being
measured, and an unknown cell name is rejected on either path.

Generated JSONL stays under gitignored `results/`. Keep the raw file local;
commit only reviewed summaries that name the model, options, engine version,
immutable model digest, GPU residency, and VRAM headroom.

## ShipOfTheseus result

Measured 2026-08-30 with Ollama 0.33.2, `phi4:14b` at digest
`ac896e5b8b34a1f4efa7b14d7520725140d5512484457fab45d2a4ea14c69dba`,
`num_ctx=8192`, one generated token, and the model reported as 100%
GPU-resident. Each measured prompt had `prompt_eval_count=1466`; the duration,
not the count, exposed reuse. Every stream contained a non-empty response chunk
and finished with `eval_count=1`.

| Invocation | Cell order | Prefix-first prompt eval | Varying-first prompt eval | Reduction |
|---:|---|---:|---:|---:|
| 1 | prefix → varying | 16.041 ms | 348.856 ms | 95.40% |
| 2 | varying → prefix | 16.278 ms | 348.162 ms | 95.32% |
| 3 | prefix → varying | 15.960 ms | 347.311 ms | 95.40% |
| **Median** | — | **16.041 ms** | **348.162 ms** | **95.39% (21.70×)** |

Median streaming TTFT (`curl time_starttransfer`) was 19.452 ms for the resumed
prefix-first request versus 407.302 ms for varying-first (95.22%, 20.94×).
Prefix-first cold TTFT was 2,464.512 ms because it includes model load; compare
the two resumed layouts for the prompt-order conclusion. Median HTTP time
through one generated token (`curl time_total`) was 19.599 ms versus 407.428 ms
(95.19%, 20.79×). Loaded-model free VRAM was 3,481–3,509 MiB, above the 2 GiB
host rule.

The ≥10% per-invocation gate passes in all three invocations. The L1 policy is
therefore to keep stable policy/tool material at the front and branch on the
request late. This result demonstrates observable Ollama cache behavior; it
does not expose a low-level cache hit rate or justify a custom L3 KV kernel.
