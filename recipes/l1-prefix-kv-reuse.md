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
- `curl`, `jq`, `nvidia-smi`, `ollama`, and `rg` are available.
- At least **2 GiB of VRAM remains free while the model is loaded**.
- The expected full prompt-token count is known. The default `1462` is pinned
  to the default model and 96-line prefix so silent context truncation fails
  closed. If either changes, first use a context known to fit comfortably,
  observe the stable full `prompt_eval_count`, then set
  `BKL_EXPECTED_PROMPT_TOKENS` to that reviewed value.

The default below uses the already-local model measured on ShipOfTheseus. Set
`BKL_MODEL` to another already-local model when reproducing elsewhere.

## Run

Run the block from the repository root. It refuses to overwrite an existing
result, stops the model between cells so every prime starts cold, alternates
cell order, exercises each cache path once before recording, and stops the
model again on exit. The prime request is excluded from the measured request.
Streaming time to first response byte is recorded as the TTFT observable.
Local API calls have configurable connect and whole-request timeouts.

```bash
set -euo pipefail

: "${BKL_MODEL:=phi4:14b}"
: "${BKL_BASE_URL:=http://127.0.0.1:11434}"
: "${BKL_NUM_CTX:=8192}"
: "${BKL_PREFIX_LINES:=96}"
: "${BKL_EXPECTED_PROMPT_TOKENS:=1462}"
: "${BKL_RUNS:=3}"
: "${BKL_MIN_FREE_MIB:=2048}"
: "${BKL_CONNECT_TIMEOUT:=5}"
: "${BKL_REQUEST_TIMEOUT:=120}"
: "${BKL_RESULT_PATH:=results/prefix-kv-reuse.jsonl}"

for tool in curl jq nvidia-smi ollama rg; do
  command -v "$tool" >/dev/null || { echo "missing required tool: $tool" >&2; exit 1; }
done

normalize_uint() {
  local name=$1
  local raw=$2
  local minimum=$3
  [[ "$raw" =~ ^[1-9][0-9]*$ ]] || {
    echo "$name must be a base-10 integer without leading zeroes" >&2
    return 1
  }
  local value
  value=$((10#$raw))
  (( value >= minimum )) || {
    echo "$name must be >= $minimum" >&2
    return 1
  }
  printf -v "$name" '%d' "$value"
}

normalize_uint BKL_RUNS "$BKL_RUNS" 3 || exit 1
normalize_uint BKL_NUM_CTX "$BKL_NUM_CTX" 1 || exit 1
normalize_uint BKL_PREFIX_LINES "$BKL_PREFIX_LINES" 1 || exit 1
normalize_uint BKL_EXPECTED_PROMPT_TOKENS "$BKL_EXPECTED_PROMPT_TOKENS" 1 || exit 1
normalize_uint BKL_MIN_FREE_MIB "$BKL_MIN_FREE_MIB" 2048 || exit 1
normalize_uint BKL_CONNECT_TIMEOUT "$BKL_CONNECT_TIMEOUT" 1 || exit 1
normalize_uint BKL_REQUEST_TIMEOUT "$BKL_REQUEST_TIMEOUT" 1 || exit 1
(( BKL_EXPECTED_PROMPT_TOKENS + 1 <= BKL_NUM_CTX )) || {
  echo "expected prompt plus one output token does not fit BKL_NUM_CTX" >&2
  exit 1
}
[[ ! -e "$BKL_RESULT_PATH" ]] || {
  echo "refusing to overwrite $BKL_RESULT_PATH" >&2
  exit 1
}

ollama show "$BKL_MODEL" >/dev/null || {
  echo "model is not already local: $BKL_MODEL (no automatic pull)" >&2
  exit 1
}
curl --silent --show-error --fail \
  --connect-timeout "$BKL_CONNECT_TIMEOUT" \
  --max-time "$BKL_REQUEST_TIMEOUT" \
  "$BKL_BASE_URL/api/tags" >/dev/null

free_mib() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' '
}

before_free_mib=$(free_mib)
(( before_free_mib >= BKL_MIN_FREE_MIB )) || {
  echo "preflight VRAM headroom is ${before_free_mib} MiB; need ${BKL_MIN_FREE_MIB} MiB" >&2
  exit 1
}

result_dir=$(dirname -- "$BKL_RESULT_PATH")
mkdir -p -- "$result_dir"
scratch_dir=$(mktemp -d "$result_dir/.bkl-prefix-kv.XXXXXX")
tmp_result="$scratch_dir/result.jsonl"

stop_model() {
  ollama stop "$BKL_MODEL" >/dev/null 2>&1 || true
  for _ in $(seq 1 30); do
    if ! ollama ps | awk 'NR > 1 { print $1 }' | rg -Fx -- "$BKL_MODEL" >/dev/null; then
      return 0
    fi
    sleep 1
  done
  echo "model did not unload within 30 seconds: $BKL_MODEL" >&2
  return 1
}

cleanup() {
  stop_model || true
  rm -rf -- "$scratch_dir"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

request() {
  local prompt=$1
  local output=$2
  local starttransfer_seconds
  starttransfer_seconds=$(jq -nc \
    --arg model "$BKL_MODEL" \
    --arg prompt "$prompt" \
    --argjson num_ctx "$BKL_NUM_CTX" \
    '{model: $model, prompt: $prompt, raw: true, stream: true,
      keep_alive: "5m",
      options: {num_ctx: $num_ctx, num_predict: 1, temperature: 0, seed: 8}}' |
    curl --silent --show-error --fail-with-body \
      --no-buffer \
      --connect-timeout "$BKL_CONNECT_TIMEOUT" \
      --max-time "$BKL_REQUEST_TIMEOUT" \
      --write-out '%{time_starttransfer}' \
      -H 'Content-Type: application/json' \
      --data-binary @- "$BKL_BASE_URL/api/generate" -o "$output")
  jq -se '
    (map(select(.done == true)) | last) as $final |
    ($final != null) and
    ($final.prompt_eval_count | type == "number") and
    ($final.prompt_eval_duration | type == "number")
  ' "$output" >/dev/null
  local prompt_eval_count
  prompt_eval_count=$(jq -s \
    'map(select(.done == true)) | last | .prompt_eval_count' "$output")
  (( prompt_eval_count == BKL_EXPECTED_PROMPT_TOKENS )) || {
    echo "prompt token mismatch: expected $BKL_EXPECTED_PROMPT_TOKENS, got $prompt_eval_count; reject possible truncation or tokenizer drift" >&2
    return 1
  }
  request_ttft_ms=$(jq -n --arg seconds "$starttransfer_seconds" \
    '$seconds | tonumber * 1000')
}

static_prefix='Reusable benchmark policy and tool schemas follow. Preserve every line verbatim.'$'\n'
for line_number in $(seq -w 1 "$BKL_PREFIX_LINES"); do
  static_prefix+="Reference clause ${line_number}: inputs are deterministic; report only the requested token."$'\n'
done

engine_version=$(ollama --version 2>&1 | head -n 1)
gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)

measure_cell() {
  local run=$1
  local cell=$2
  local prime_prompt measured_prompt
  local prime_json="$scratch_dir/prime.json"
  local measured_json="$scratch_dir/measured.json"

  case "$cell" in
    prefix_first)
      prime_prompt="$static_prefix"$'\n''Alpha request: return the digit zero.'
      measured_prompt="$static_prefix"$'\n''Beta request: return the digit zero.'
      ;;
    varying_first)
      prime_prompt='Alpha request: return the digit zero.'$'\n'"$static_prefix"
      measured_prompt='Beta request: return the digit zero.'$'\n'"$static_prefix"
      ;;
    *)
      echo "unknown cell: $cell" >&2
      return 1
      ;;
  esac

  stop_model
  request "$prime_prompt" "$prime_json"
  local prime_ttft_ms
  prime_ttft_ms=$request_ttft_ms

  local started_ns finished_ns
  started_ns=$(date +%s%N)
  request "$measured_prompt" "$measured_json"
  local resume_ttft_ms
  resume_ttft_ms=$request_ttft_ms
  finished_ns=$(date +%s%N)

  local loaded_free_mib
  loaded_free_mib=$(free_mib)
  (( loaded_free_mib >= BKL_MIN_FREE_MIB )) || {
    echo "loaded-model VRAM headroom is ${loaded_free_mib} MiB; need ${BKL_MIN_FREE_MIB} MiB" >&2
    return 1
  }

  local processor model_row
  model_row=$(ollama ps | awk -v model="$BKL_MODEL" '$1 == model { print; exit }')
  processor=$(rg -o '[0-9]+% GPU' <<<"$model_row" | head -n 1 || true)
  [[ "$processor" == '100% GPU' ]] || {
    echo "model is not 100% GPU-resident: ${model_row:-missing from ollama ps}" >&2
    return 1
  }

  jq -nc \
    --arg timestamp "$(date --utc +%Y-%m-%dT%H:%M:%SZ)" \
    --arg engine_version "$engine_version" \
    --arg model "$BKL_MODEL" \
    --arg gpu "$gpu_name" \
    --arg driver "$driver_version" \
    --arg processor "$processor" \
    --arg cell "$cell" \
    --argjson run "$run" \
    --argjson num_ctx "$BKL_NUM_CTX" \
    --argjson prefix_lines "$BKL_PREFIX_LINES" \
    --argjson expected_prompt_tokens "$BKL_EXPECTED_PROMPT_TOKENS" \
    --argjson connect_timeout_seconds "$BKL_CONNECT_TIMEOUT" \
    --argjson request_timeout_seconds "$BKL_REQUEST_TIMEOUT" \
    --argjson prompt_eval_count "$(jq -s 'map(select(.done == true)) | last | .prompt_eval_count' "$measured_json")" \
    --argjson prompt_eval_ms "$(jq -s 'map(select(.done == true)) | last | .prompt_eval_duration / 1000000' "$measured_json")" \
    --argjson one_token_total_ms "$((finished_ns - started_ns))" \
    --argjson prime_prompt_eval_ms "$(jq -s 'map(select(.done == true)) | last | .prompt_eval_duration / 1000000' "$prime_json")" \
    --argjson prime_ttft_ms "$prime_ttft_ms" \
    --argjson resume_ttft_ms "$resume_ttft_ms" \
    --argjson vram_free_mib "$loaded_free_mib" \
    '{schema_version: 1, timestamp: $timestamp,
      engine: {name: "ollama", version: $engine_version},
      model: $model, gpu: $gpu, driver: $driver, processor: $processor,
      options: {num_ctx: $num_ctx, num_predict: 1, temperature: 0, seed: 8,
                prefix_lines: $prefix_lines,
                expected_prompt_tokens: $expected_prompt_tokens,
                connect_timeout_seconds: $connect_timeout_seconds,
                request_timeout_seconds: $request_timeout_seconds},
      run: $run, cell: $cell,
      ttft_method: "curl time_starttransfer on the streaming response",
      prompt_eval_count: $prompt_eval_count,
      prompt_eval_ms: $prompt_eval_ms,
      one_token_total_ms: ($one_token_total_ms / 1000000),
      prime_prompt_eval_ms: $prime_prompt_eval_ms,
      prime_ttft_ms: $prime_ttft_ms,
      resume_ttft_ms: $resume_ttft_ms,
      vram_free_mib: $vram_free_mib}' >>"$tmp_result"
}

warmup_cell() {
  local cell=$1
  local prime_prompt measured_prompt
  local warmup_prime_json="$scratch_dir/warmup-prime.json"
  local warmup_measured_json="$scratch_dir/warmup-measured.json"

  case "$cell" in
    prefix_first)
      prime_prompt="$static_prefix"$'\n''Alpha request: return the digit zero.'
      measured_prompt="$static_prefix"$'\n''Beta request: return the digit zero.'
      ;;
    varying_first)
      prime_prompt='Alpha request: return the digit zero.'$'\n'"$static_prefix"
      measured_prompt='Beta request: return the digit zero.'$'\n'"$static_prefix"
      ;;
  esac

  stop_model
  request "$prime_prompt" "$warmup_prime_json"
  request "$measured_prompt" "$warmup_measured_json"
}

# Exercise both cache paths once outside the recorded invocations. On this
# host, the engine's first-ever prefix-reuse request can include one-time
# initialization even after its prime request has returned.
warmup_cell prefix_first
warmup_cell varying_first
stop_model

for run in $(seq 1 "$BKL_RUNS"); do
  if (( run % 2 == 0 )); then
    measure_cell "$run" varying_first
    measure_cell "$run" prefix_first
  else
    measure_cell "$run" prefix_first
    measure_cell "$run" varying_first
  fi
done

mv -- "$tmp_result" "$BKL_RESULT_PATH"

summary=$(jq -s '
  def median:
    sort as $s | length as $n |
    if ($n % 2) == 1 then $s[($n / 2 | floor)]
    else (($s[$n / 2 - 1] + $s[$n / 2]) / 2) end;
  (map(select(.cell == "prefix_first") | .prompt_eval_ms) | median) as $prefix_median |
  (map(select(.cell == "varying_first") | .prompt_eval_ms) | median) as $varying_median |
  (map(select(.cell == "prefix_first") | .prime_ttft_ms) | median) as $prefix_cold_ttft |
  (map(select(.cell == "prefix_first") | .resume_ttft_ms) | median) as $prefix_resume_ttft |
  (map(select(.cell == "varying_first") | .resume_ttft_ms) | median) as $varying_resume_ttft |
  (group_by(.run) | map(
    (map(select(.cell == "prefix_first"))[0]) as $prefix |
    (map(select(.cell == "varying_first"))[0]) as $varying |
    {run: $prefix.run,
     same_prompt_tokens: ($prefix.prompt_eval_count == $varying.prompt_eval_count),
     reduction_pct: (100 * (1 - ($prefix.prompt_eval_ms / $varying.prompt_eval_ms)))}
  )) as $runs |
  {prefix_first_median_ms: $prefix_median,
   varying_first_median_ms: $varying_median,
   median_reduction_pct: (100 * (1 - ($prefix_median / $varying_median))),
   speedup_x: ($varying_median / $prefix_median),
   prefix_first_cold_ttft_ms: $prefix_cold_ttft,
   prefix_first_resume_ttft_ms: $prefix_resume_ttft,
   varying_first_resume_ttft_ms: $varying_resume_ttft,
   resume_ttft_reduction_pct: (100 * (1 - ($prefix_resume_ttft / $varying_resume_ttft))),
   per_run: $runs,
   decision_gate: (if all($runs[]; .same_prompt_tokens and .reduction_pct >= 10)
                   then "pass" else "no-go" end)}
' "$BKL_RESULT_PATH")

jq . <<<"$summary"
jq -e '.decision_gate == "pass"' <<<"$summary" >/dev/null
```

Generated JSONL stays under gitignored `results/`. Keep the raw file local;
commit only reviewed summaries that name the model, options, engine version,
GPU residency, and VRAM headroom.

## ShipOfTheseus result

Measured 2026-08-30 with Ollama 0.33.2, `phi4:14b`, `num_ctx=8192`, one
generated token, and the model reported as 100% GPU-resident. Each measured
prompt had `prompt_eval_count=1462`; the duration, not the count, exposed reuse.

| Invocation | Cell order | Prefix-first prompt eval | Varying-first prompt eval | Reduction |
|---:|---|---:|---:|---:|
| 1 | prefix → varying | 24.931 ms | 351.043 ms | 92.90% |
| 2 | varying → prefix | 21.951 ms | 355.731 ms | 93.83% |
| 3 | prefix → varying | 23.369 ms | 349.777 ms | 93.32% |
| **Median** | — | **23.369 ms** | **351.043 ms** | **93.34% (15.02×)** |

Median streaming TTFT (`curl time_starttransfer`) was 26.994 ms for the resumed
prefix-first request versus 411.945 ms for varying-first (93.45%, 15.26×).
Prefix-first cold TTFT was 2,363.259 ms because it includes model load; compare
the two resumed layouts for the prompt-order conclusion. Median outer request
time through one generated token was 35.870 ms versus 422.360 ms (91.51%,
11.77×). Loaded-model free VRAM was 3,428–3,477 MiB, above the 2 GiB host rule.

The ≥10% per-invocation gate passes in all three invocations. The L1 policy is
therefore to keep stable policy/tool material at the front and branch on the
request late. This result demonstrates observable Ollama cache behavior; it
does not expose a low-level cache hit rate or justify a custom L3 KV kernel.
