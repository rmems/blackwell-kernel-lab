#!/usr/bin/env bash
# L1 kernel ablation cells A–C on llama-server (#16 / RM-470).
# Usage: from repo root, with a GGUF path:
#   MODEL=/path/to/model.gguf bash harness/serve/run_l1_ablation.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

MODEL="${MODEL:-}"
if [[ -z "$MODEL" ]]; then
  echo "Set MODEL=/path/to.gguf" >&2
  exit 2
fi
PORT="${PORT:-8080}"
HOST="${HOST:-127.0.0.1}"
NGL="${NGL:-99}"
CTX="${CTX:-4096}"
OUT="${OUT:-results}"
ENGINE_VER="${ENGINE_VER:-$(llama-server --version 2>&1 | head -1)}"
API_MODEL=""  # filled after first /v1/models
SPID=""

kill_server() {
  # Prefer the child we started.
  if [[ -n "${SPID:-}" ]] && kill -0 "$SPID" 2>/dev/null; then
    kill "$SPID" 2>/dev/null || true
    wait "$SPID" 2>/dev/null || true
  fi
  # Fallback: only kill a listener that is actually llama-server on this port.
  # Never wait() on foreign PIDs (not our children). Never kill by port alone.
  local pid comm
  while read -r pid; do
    [[ -z "${pid:-}" ]] && continue
    comm=$(ps -p "$pid" -o comm= 2>/dev/null || true)
    if [[ "$comm" == *llama-server* ]]; then
      kill "$pid" 2>/dev/null || true
    fi
  done < <(
    ss -ltnp "sport = :${PORT}" 2>/dev/null \
      | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' \
      | sort -u
  )
  sleep 1
  SPID=""
}

start_server() {
  local fa="$1"
  local log="/tmp/bkl-llama-fa-${fa}.log"
  kill_server
  echo ">> starting llama-server -fa ${fa}"
  llama-server -m "$MODEL" -ngl "$NGL" -c "$CTX" --port "$PORT" --host "$HOST" -fa "$fa" \
    >"$log" 2>&1 &
  SPID=$!
  trap kill_server EXIT
  for i in $(seq 1 120); do
    if curl -sf "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; then
      API_MODEL=$(curl -sS "http://${HOST}:${PORT}/v1/models" | python3 -c \
        'import sys,json; d=json.load(sys.stdin); print(d["data"][0]["id"] if d.get("data") else "")')
      echo ">> ready model=${API_MODEL} (${i}s)"
      return 0
    fi
    sleep 1
  done
  echo "server failed to start; log:" >&2
  tail -40 "$log" >&2
  exit 1
}

run_cell() {
  local label="$1"
  local flags="$2"
  local quant="${3:-UD-Q4_K_XL}"
  # warm
  python3 harness/serve/smoke_openai.py --out "$OUT" --stream --skip-content-slo \
    --base-url "http://${HOST}:${PORT}/v1" --model "$API_MODEL" \
    --label "${label}-warmup" --engine-version "$ENGINE_VER" --engine-flags "$flags" \
    --quant "$quant" --max-tokens 16 >/dev/null || true
  # measured
  python3 harness/serve/smoke_openai.py --out "$OUT" --stream --skip-content-slo \
    --base-url "http://${HOST}:${PORT}/v1" --model "$API_MODEL" \
    --label "$label" --engine-version "$ENGINE_VER" --engine-flags "$flags" \
    --quant "$quant" --max-tokens 64 --agent-prompt
}

COMMON="-ngl ${NGL} -c ${CTX}"

# Cell A: FA off
start_server off
run_cell "cell-A-fa-off" "${COMMON} -fa off"

# Cell B: FA on
start_server on
run_cell "cell-B-fa-on" "${COMMON} -fa on"

# Cell C: FA on + graphs note (no CLI toggle on llama.cpp 9190 — engine default)
# Reuse FA-on server; second warm-measured pair labeled as graphs=default
run_cell "cell-C-fa-on-graphs-default" "${COMMON} -fa on (cuda_graphs=engine-default no CLI toggle on 9190)"

kill_server
trap - EXIT
echo ">> done. Aggregate: python3 harness/report/aggregate.py --results ${OUT}/"
python3 harness/report/aggregate.py --results "$OUT/" || true
