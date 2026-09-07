#!/usr/bin/env bash
#
# L1 prefix / KV reuse measurement on sm_120.
#
# Compares two prompt layouts against a locally resident model behind Ollama:
#
#   prefix_first   [stable policy + tool schemas][varying task]
#   varying_first  [varying task][stable policy + tool schemas]
#
# Refuses to overwrite an existing result, stops the model between cells so
# every prime starts cold, alternates cell order, exercises each cache path once
# before recording, and stops the model again on exit. The prime request is
# excluded from the measured request.
#
# Method, policy, prerequisites and the reviewed ShipOfTheseus numbers live in
# ../l1-prefix-kv-reuse.md. Raw JSONL is written under gitignored results/.
#
# Configuration is by environment variable; only the result basename is
# configurable, and raw output always stays under results/.

set -euo pipefail

: "${BKL_MODEL:=phi4:14b}"
: "${BKL_BASE_URL:=http://127.0.0.1:11434}"
: "${BKL_NUM_CTX:=8192}"
: "${BKL_PREFIX_LINES:=96}"
: "${BKL_EXPECTED_PROMPT_TOKENS:=1466}"
: "${BKL_RUNS:=3}"
: "${BKL_MIN_FREE_MIB:=2048}"
: "${BKL_CONNECT_TIMEOUT:=5}"
: "${BKL_REQUEST_TIMEOUT:=120}"
: "${BKL_RESULT_NAME:=prefix-kv-reuse.jsonl}"

for tool in curl flock jq ln nvidia-smi ollama rg sha256sum; do
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
[[ "$BKL_BASE_URL" =~ ^http://127\.0\.0\.1(:[0-9]+)?$ ]] || {
  echo "BKL_BASE_URL must be a canonical IPv4-loopback Ollama origin" >&2
  exit 1
}
[[ "$BKL_RESULT_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*\.jsonl$ ]] || {
  echo "BKL_RESULT_NAME must be a .jsonl basename without directories" >&2
  exit 1
}
BKL_RESULT_PATH="results/$BKL_RESULT_NAME"
export OLLAMA_HOST="$BKL_BASE_URL"
[[ ! -e "$BKL_RESULT_PATH" ]] || {
  echo "refusing to overwrite $BKL_RESULT_PATH" >&2
  exit 1
}

mkdir -p -- results
lock_key=$(printf '%s' "$BKL_BASE_URL" |
  sha256sum | awk '{ print $1 }')
lock_path="results/.prefix-kv-reuse-$lock_key.lock"
exec {BKL_LOCK_FD}>"$lock_path"
flock -n "$BKL_LOCK_FD" || {
  echo "another prefix/KV run is using $BKL_MODEL at $BKL_BASE_URL" >&2
  exit 1
}

ollama show "$BKL_MODEL" >/dev/null || {
  echo "model is not already local: $BKL_MODEL (no automatic pull)" >&2
  exit 1
}
tags_json=$(curl --silent --show-error --fail \
  --connect-timeout "$BKL_CONNECT_TIMEOUT" \
  --max-time "$BKL_REQUEST_TIMEOUT" \
  "$BKL_BASE_URL/api/tags")
model_digest=$(jq -er --arg model "$BKL_MODEL" '
  first(.models[] |
    select((.name == $model) or (.model == $model)) |
    .digest) // empty
' <<<"$tags_json") || {
  echo "could not resolve an immutable digest for local model: $BKL_MODEL" >&2
  exit 1
}
[[ "$model_digest" =~ ^[0-9a-f]{64}$ ]] || {
  echo "invalid model digest for $BKL_MODEL: $model_digest" >&2
  exit 1
}

free_mib() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' '
}

assert_loaded_model() {
  BKL_LAST_FREE_MIB=$(free_mib)
  (( BKL_LAST_FREE_MIB >= BKL_MIN_FREE_MIB )) || {
    echo "loaded-model VRAM headroom is ${BKL_LAST_FREE_MIB} MiB; need ${BKL_MIN_FREE_MIB} MiB" >&2
    return 1
  }

  local ps_output
  if ! ps_output=$(ollama ps 2>&1); then
    echo "ollama ps failed while validating model residency: $ps_output" >&2
    return 1
  fi
  BKL_LAST_MODEL_ROW=$(awk -v model="$BKL_MODEL" '$1 == model { print; exit }' <<<"$ps_output")
  BKL_LAST_PROCESSOR=$(rg -o '[0-9]+% GPU' <<<"$BKL_LAST_MODEL_ROW" | head -n 1 || true)
  [[ "$BKL_LAST_PROCESSOR" == '100% GPU' ]] || {
    echo "model is not 100% GPU-resident: ${BKL_LAST_MODEL_ROW:-missing from ollama ps}" >&2
    return 1
  }
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
  local stop_output
  if ! stop_output=$(ollama stop "$BKL_MODEL" 2>&1); then
    echo "ollama stop failed: $stop_output" >&2
    return 1
  fi
  for _ in $(seq 1 30); do
    local ps_output loaded_name
    if ! ps_output=$(ollama ps 2>&1); then
      echo "ollama ps failed while confirming unload: $ps_output" >&2
      return 1
    fi
    loaded_name=$(awk -v model="$BKL_MODEL" '$1 == model { print $1; exit }' <<<"$ps_output")
    if [[ -z "$loaded_name" ]]; then
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
  local curl_timings
  curl_timings=$(jq -nc \
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
      --write-out '{"time_starttransfer":%{time_starttransfer},"time_total":%{time_total}}' \
      -H 'Content-Type: application/json' \
      --data-binary @- "$BKL_BASE_URL/api/generate" -o "$output")
  jq -se '
    . as $events |
    ($events | map(select(.done == true)) | last) as $final |
    ($final != null) and
    ($final.prompt_eval_count | type == "number") and
    ($final.prompt_eval_duration | type == "number") and
    ($final.eval_count == 1) and
    ([$events[] | .response? // empty |
      select((type == "string") and (length > 0))] | length > 0)
  ' "$output" >/dev/null || {
    echo "generation must stream exactly one response token" >&2
    return 1
  }
  local prompt_eval_count
  prompt_eval_count=$(jq -s \
    'map(select(.done == true)) | last | .prompt_eval_count' "$output")
  (( prompt_eval_count == BKL_EXPECTED_PROMPT_TOKENS )) || {
    echo "prompt token mismatch: expected $BKL_EXPECTED_PROMPT_TOKENS, got $prompt_eval_count; reject possible truncation or tokenizer drift" >&2
    return 1
  }
  request_ttft_ms=$(jq -er '.time_starttransfer * 1000' <<<"$curl_timings")
  request_total_ms=$(jq -er '.time_total * 1000' <<<"$curl_timings")
}

static_prefix='Reusable benchmark policy and tool schemas follow. Preserve every line verbatim.'$'\n'
for line_number in $(seq -w 1 "$BKL_PREFIX_LINES"); do
  static_prefix+="Reference clause ${line_number}: inputs are deterministic; report only the requested token."$'\n'
done
static_prefix_sha256=$(printf '%s' "$static_prefix" | sha256sum | awk '{ print $1 }')
response_cue=$'\n''Assistant response: ['

engine_version=$(ollama --version 2>&1 | head -n 1)
gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)

# The recorded pass and the warmup pass must send byte-identical prompts for a
# cell, or the warmup exercises a different cache path than the one measured.
# Both call this, so the two cannot drift, and an unknown cell is rejected on
# either path rather than silently proceeding with empty prompts.
build_cell_prompts() {
  local cell=$1
  case "$cell" in
    prefix_first)
      BKL_PRIME_PROMPT="$static_prefix"$'\n''Alpha request: return the digit zero.'"$response_cue"
      BKL_MEASURED_PROMPT="$static_prefix"$'\n''Beta request: return the digit zero.'"$response_cue"
      ;;
    varying_first)
      BKL_PRIME_PROMPT='Alpha request: return the digit zero.'$'\n'"$static_prefix$response_cue"
      BKL_MEASURED_PROMPT='Beta request: return the digit zero.'$'\n'"$static_prefix$response_cue"
      ;;
    *)
      echo "unknown cell: $cell" >&2
      return 1
      ;;
  esac
}

measure_cell() {
  local run=$1
  local cell=$2
  local prime_json="$scratch_dir/prime.json"
  local measured_json="$scratch_dir/measured.json"

  build_cell_prompts "$cell"
  local prime_prompt=$BKL_PRIME_PROMPT
  local measured_prompt=$BKL_MEASURED_PROMPT

  local prompt_pair_sha256
  prompt_pair_sha256=$(printf '%s\0%s' "$prime_prompt" "$measured_prompt" |
    sha256sum | awk '{ print $1 }')

  stop_model
  request "$prime_prompt" "$prime_json"
  local prime_ttft_ms
  prime_ttft_ms=$request_ttft_ms
  assert_loaded_model
  local vram_free_mib_before_resume
  vram_free_mib_before_resume=$BKL_LAST_FREE_MIB

  request "$measured_prompt" "$measured_json"
  local resume_ttft_ms measured_total_ms
  resume_ttft_ms=$request_ttft_ms
  measured_total_ms=$request_total_ms
  assert_loaded_model
  local loaded_free_mib processor
  loaded_free_mib=$BKL_LAST_FREE_MIB
  processor=$BKL_LAST_PROCESSOR

  jq -nc \
    --arg timestamp "$(date --utc +%Y-%m-%dT%H:%M:%SZ)" \
    --arg engine_version "$engine_version" \
    --arg model "$BKL_MODEL" \
    --arg model_digest "$model_digest" \
    --arg gpu "$gpu_name" \
    --arg driver "$driver_version" \
    --arg processor "$processor" \
    --arg endpoint "$BKL_BASE_URL" \
    --arg static_prefix_sha256 "$static_prefix_sha256" \
    --arg prompt_pair_sha256 "$prompt_pair_sha256" \
    --arg cell "$cell" \
    --argjson run "$run" \
    --argjson num_ctx "$BKL_NUM_CTX" \
    --argjson prefix_lines "$BKL_PREFIX_LINES" \
    --argjson expected_prompt_tokens "$BKL_EXPECTED_PROMPT_TOKENS" \
    --argjson connect_timeout_seconds "$BKL_CONNECT_TIMEOUT" \
    --argjson request_timeout_seconds "$BKL_REQUEST_TIMEOUT" \
    --argjson vram_free_mib_before_resume "$vram_free_mib_before_resume" \
    --argjson prompt_eval_count "$(jq -s 'map(select(.done == true)) | last | .prompt_eval_count' "$measured_json")" \
    --argjson eval_count "$(jq -s 'map(select(.done == true)) | last | .eval_count' "$measured_json")" \
    --argjson prompt_eval_ms "$(jq -s 'map(select(.done == true)) | last | .prompt_eval_duration / 1000000' "$measured_json")" \
    --argjson one_token_total_ms "$measured_total_ms" \
    --argjson prime_prompt_eval_ms "$(jq -s 'map(select(.done == true)) | last | .prompt_eval_duration / 1000000' "$prime_json")" \
    --argjson prime_ttft_ms "$prime_ttft_ms" \
    --argjson resume_ttft_ms "$resume_ttft_ms" \
    --argjson vram_free_mib "$loaded_free_mib" \
    '{schema_version: 1, timestamp: $timestamp,
      engine: {name: "ollama", version: $engine_version, endpoint: $endpoint},
      model: $model, model_digest: $model_digest,
      gpu: $gpu, driver: $driver, processor: $processor,
      static_prefix_sha256: $static_prefix_sha256,
      prompt_pair_sha256: $prompt_pair_sha256,
      options: {num_ctx: $num_ctx, num_predict: 1, temperature: 0, seed: 8,
                prefix_lines: $prefix_lines,
                expected_prompt_tokens: $expected_prompt_tokens,
                connect_timeout_seconds: $connect_timeout_seconds,
                request_timeout_seconds: $request_timeout_seconds},
      run: $run, cell: $cell,
      ttft_method: "curl time_starttransfer on the streaming response",
      prompt_eval_count: $prompt_eval_count,
      eval_count: $eval_count,
      prompt_eval_ms: $prompt_eval_ms,
      one_token_total_ms: $one_token_total_ms,
      prime_prompt_eval_ms: $prime_prompt_eval_ms,
      prime_ttft_ms: $prime_ttft_ms,
      resume_ttft_ms: $resume_ttft_ms,
      vram_free_mib_before_resume: $vram_free_mib_before_resume,
      vram_free_mib: $vram_free_mib}' >>"$tmp_result"
}

warmup_cell() {
  local cell=$1
  local warmup_prime_json="$scratch_dir/warmup-prime.json"
  local warmup_measured_json="$scratch_dir/warmup-measured.json"

  build_cell_prompts "$cell"
  local prime_prompt=$BKL_PRIME_PROMPT
  local measured_prompt=$BKL_MEASURED_PROMPT

  stop_model
  request "$prime_prompt" "$warmup_prime_json"
  assert_loaded_model
  request "$measured_prompt" "$warmup_measured_json"
  assert_loaded_model
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

ln -- "$tmp_result" "$BKL_RESULT_PATH" || {
  echo "refusing to overwrite concurrently created $BKL_RESULT_PATH" >&2
  exit 1
}
rm -- "$tmp_result"

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
