#!/usr/bin/env bash
#
# Run NVIDIA Compute Sanitizer on first-party CUDA binaries.
#
# Used by .github/workflows/ci-gpu-sanitizer.yml and
# recipes/compute-sanitizer.md. CPU tests exercise this file with fake
# tools; they do not need a GPU.
#
# Usage:
#   kernels/tools/run_compute_sanitizer.sh --suite \
#       --bin-dir build/kernels/src --out-dir results/sanitizer
#   kernels/tools/run_compute_sanitizer.sh --tool memcheck --out-dir DIR -- \
#       ./build/kernels/src/bkl_device_hello
#
set -euo pipefail

SCHEMA=bkl.compute_sanitizer.v1
SUITE_SCHEMA=bkl.compute_sanitizer.suite.v1
PRINT_LIMIT=50
ERROR_EXITCODE=1
PER_JOB_TIMEOUT_S=900
PREVIEW_BYTES=65536
CUDA_BIN=/usr/local/cuda/bin
MAX_LOG_BYTES=1048576
SUITE=0
BIN_DIR=
OUT_DIR=
TOOL=memcheck
APP=()
FAILURES=()
RUN_COUNT=0

usage() {
  echo "usage: $0 --suite --bin-dir DIR --out-dir DIR" >&2
  echo "       $0 --tool TOOL --out-dir DIR [--max-log-bytes N] -- BINARY [ARGS...]" >&2
  exit 2
}

find_cuda_tool() {
  local name=$1
  local env_key env_val
  env_key=$(printf '%s' "$name" | tr '-' '_' | tr '[:lower:]' '[:upper:]')
  env_val=${!env_key-}
  if [ -n "$env_val" ]; then
    printf '%s\n' "$env_val"
    return 0
  fi
  if command -v "$name" >/dev/null 2>&1; then
    command -v "$name"
    return 0
  fi
  if [ -x "${CUDA_BIN}/${name}" ]; then
    printf '%s\n' "${CUDA_BIN}/${name}"
    return 0
  fi
  return 1
}

capture_bound() {
  local limit=$1
  shift
  local text
  text=$("$@" 2>&1) || true
  text=${text%"${text##*[![:space:]]}"}
  if [ "${#text}" -gt "$limit" ]; then
    printf '%s\n' "${text:0:limit}"
    echo "[truncated]"
  else
    printf '%s\n' "$text"
  fi
}

csv_field() {
  local line=$1
  local index=$2
  local field
  field=$(printf '%s\n' "$line" | cut -d, -f"$index")
  field=${field#"${field%%[![:space:]]*}"}
  field=${field%"${field##*[![:space:]]}"}
  printf '%s' "$field"
}

parse_error_summary() {
  local log=$1
  local line
  line=$(grep -E 'ERROR SUMMARY:[[:space:]]+[0-9]+[[:space:]]+errors?' "$log" | tail -n 1 || true)
  if [ -z "$line" ]; then
    return 1
  fi
  printf '%s\n' "$line" | grep -Eo '[0-9]+' | tail -n 1
}

maybe_truncate() {
  local path=$1
  local max=$2
  local size keep
  size=$(wc -c < "$path")
  if [ "$size" -le "$max" ]; then
    echo 0
    return 0
  fi
  keep=$((max - 32))
  if [ "$keep" -lt 0 ]; then
    keep=0
  fi
  head -c "$keep" "$path" > "${path}.tmp"
  printf '\n[truncated]\n' >> "${path}.tmp"
  mv "${path}.tmp" "$path"
  echo 1
}

preview_log() {
  local path=$1
  local size
  head -c "$PREVIEW_BYTES" "$path" || true
  size=$(wc -c < "$path")
  if [ "$size" -gt "$PREVIEW_BYTES" ]; then
    echo
    echo "[job-log preview truncated]"
  fi
}

json_list() {
  python3 -c 'import json,sys; print(json.dumps(sys.argv[1:]))' -- "$@"
}

write_meta() {
  python3 - <<'PY'
import json
import os
from pathlib import Path

def maybe_int(value: str):
    if value in ("", "null"):
        return None
    return int(value)

payload = {
    "schema": os.environ["BKL_SCHEMA"],
    "tool": os.environ["BKL_TOOL"],
    "sanitizer_version": os.environ.get("BKL_SANITIZER_VERSION") or None,
    "cuda": {"nvcc_version": os.environ.get("BKL_NVCC_VERSION") or None},
    "driver_version": os.environ.get("BKL_DRIVER_VERSION") or None,
    "gpu": {
        "name": os.environ.get("BKL_GPU_NAME") or None,
        "compute_cap": os.environ.get("BKL_COMPUTE_CAP") or None,
    },
    "binary": {
        "path": os.environ["BKL_BINARY_PATH"],
        "sha256": os.environ["BKL_BINARY_SHA256"],
    },
    "command": json.loads(os.environ["BKL_COMMAND_JSON"]),
    "exit_code": int(os.environ["BKL_EXIT_CODE"]),
    "error_summary": maybe_int(os.environ.get("BKL_ERROR_SUMMARY", "")),
    "log_path": os.environ["BKL_LOG_PATH"],
    "log_bytes": int(os.environ["BKL_LOG_BYTES"]),
    "log_truncated": os.environ["BKL_LOG_TRUNCATED"] == "1",
}
Path(os.environ["BKL_META_PATH"]).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
PY
}

append_summary() {
  python3 - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["BKL_SUMMARY_PATH"])
if path.is_file():
    data = json.loads(path.read_text())
else:
    data = {
        "schema": os.environ["BKL_SUITE_SCHEMA"],
        "runs": [],
        "failures": [],
    }
if os.environ.get("BKL_RUN_META"):
    data["runs"].append(json.loads(Path(os.environ["BKL_RUN_META"]).read_text()))
if os.environ.get("BKL_FAILURE"):
    data["failures"].append(os.environ["BKL_FAILURE"])
path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
PY
}

run_one() {
  local tool=$1
  local binary=$2
  shift 2
  local stem log_path meta_path status error_summary truncated digest size
  local -a cmd

  case "$tool" in
    memcheck | initcheck | racecheck | synccheck) ;;
    *)
      echo "unsupported sanitizer tool: $tool" >&2
      return 1
      ;;
  esac
  if [ ! -f "$binary" ]; then
    echo "missing binary: $binary" >&2
    return 1
  fi

  mkdir -p "$OUT_DIR"
  stem="${tool}-$(basename "$binary")"
  log_path="${OUT_DIR}/${stem}.log"
  meta_path="${OUT_DIR}/${stem}.json"
  cmd=(
    "$SANITIZER"
    --tool "$tool"
    --error-exitcode "$ERROR_EXITCODE"
    --print-limit "$PRINT_LIMIT"
    --check-exit-code yes
    "$binary"
    "$@"
  )

  set +e
  if command -v timeout >/dev/null 2>&1; then
    timeout --signal=TERM "$PER_JOB_TIMEOUT_S" "${cmd[@]}" >"$log_path" 2>&1
    status=$?
  else
    "${cmd[@]}" >"$log_path" 2>&1
    status=$?
  fi
  set -e

  error_summary=
  if error_summary=$(parse_error_summary "$log_path"); then
    :
  else
    error_summary=
  fi
  truncated=$(maybe_truncate "$log_path" "$MAX_LOG_BYTES")
  digest=$(sha256sum "$binary" | awk '{print $1}')
  size=$(wc -c < "$log_path")

  BKL_SCHEMA=$SCHEMA \
    BKL_TOOL=$tool \
    BKL_SANITIZER_VERSION=$SANITIZER_VERSION \
    BKL_NVCC_VERSION=$NVCC_VERSION \
    BKL_DRIVER_VERSION=$DRIVER_VERSION \
    BKL_GPU_NAME=$GPU_NAME \
    BKL_COMPUTE_CAP=$COMPUTE_CAP \
    BKL_BINARY_PATH=$binary \
    BKL_BINARY_SHA256=$digest \
    BKL_COMMAND_JSON=$(json_list "${cmd[@]}") \
    BKL_EXIT_CODE=$status \
    BKL_ERROR_SUMMARY=${error_summary:-} \
    BKL_LOG_PATH=$log_path \
    BKL_LOG_BYTES=$size \
    BKL_LOG_TRUNCATED=$truncated \
    BKL_META_PATH=$meta_path \
    write_meta

  echo "wrote ${meta_path}"
  preview_log "$log_path"
  RUN_COUNT=$((RUN_COUNT + 1))

  BKL_SUMMARY_PATH="${OUT_DIR}/summary.json" \
    BKL_SUITE_SCHEMA=$SUITE_SCHEMA \
    BKL_RUN_META=$meta_path \
    BKL_FAILURE='' \
    append_summary

  if [ -z "$error_summary" ]; then
    echo "${log_path}: missing ERROR SUMMARY" >&2
    FAILURES+=("${log_path}: missing ERROR SUMMARY")
    BKL_SUMMARY_PATH="${OUT_DIR}/summary.json" \
      BKL_SUITE_SCHEMA=$SUITE_SCHEMA \
      BKL_RUN_META='' \
      BKL_FAILURE="${log_path}: missing ERROR SUMMARY" \
      append_summary
    return 1
  fi
  if [ "$error_summary" != 0 ] || [ "$status" -ne 0 ]; then
    echo "${tool} $(basename "$binary"): exit_code=${status} error_summary=${error_summary}" >&2
    FAILURES+=("${tool} $(basename "$binary"): exit_code=${status} error_summary=${error_summary}")
    BKL_SUMMARY_PATH="${OUT_DIR}/summary.json" \
      BKL_SUITE_SCHEMA=$SUITE_SCHEMA \
      BKL_RUN_META='' \
      BKL_FAILURE="${tool} $(basename "$binary"): exit_code=${status} error_summary=${error_summary}" \
      append_summary
    return 1
  fi
  return 0
}

run_suite() {
  run_one memcheck "${BIN_DIR}/bkl_device_hello" || true
  run_one initcheck "${BIN_DIR}/bkl_device_hello" || true
  run_one memcheck "${BIN_DIR}/bkl_graph_launch_bench" \
    --smoke --out "${OUT_DIR}/graph-launch-memcheck.json" || true
  run_one initcheck "${BIN_DIR}/bkl_graph_launch_bench" \
    --smoke --out "${OUT_DIR}/graph-launch-initcheck.json" || true
  run_one memcheck "${BIN_DIR}/bkl_green_ctx_bench" \
    --smoke --out "${OUT_DIR}/green-ctx-memcheck.json" || true
}

while [ $# -gt 0 ]; do
  case "$1" in
    --suite)
      SUITE=1
      shift
      ;;
    --bin-dir)
      [ $# -ge 2 ] || usage
      BIN_DIR=$2
      shift 2
      ;;
    --out-dir)
      [ $# -ge 2 ] || usage
      OUT_DIR=$2
      shift 2
      ;;
    --tool)
      [ $# -ge 2 ] || usage
      TOOL=$2
      shift 2
      ;;
    --max-log-bytes)
      [ $# -ge 2 ] || usage
      MAX_LOG_BYTES=$2
      shift 2
      ;;
    --)
      shift
      APP=("$@")
      break
      ;;
    --*)
      usage
      ;;
    *)
      APP=("$@")
      break
      ;;
  esac
done

[ -n "$OUT_DIR" ] || usage
mkdir -p "$OUT_DIR"

SANITIZER=$(find_cuda_tool compute-sanitizer) || {
  echo "compute-sanitizer not found on PATH or ${CUDA_BIN}" >&2
  exit 1
}
NVCC=$(find_cuda_tool nvcc || true)
NVIDIA_SMI=$(find_cuda_tool nvidia-smi || true)

SANITIZER_VERSION=$(capture_bound 512 "$SANITIZER" --version)
SANITIZER_VERSION=${SANITIZER_VERSION%$'\n'}
if [ -z "$SANITIZER_VERSION" ]; then
  echo "compute-sanitizer --version produced no output" >&2
  exit 1
fi
if [ -n "$NVCC" ]; then
  NVCC_VERSION=$(capture_bound 512 "$NVCC" --version)
  NVCC_VERSION=${NVCC_VERSION%$'\n'}
else
  NVCC_VERSION=
fi

GPU_NAME=
DRIVER_VERSION=
COMPUTE_CAP=
if [ -n "$NVIDIA_SMI" ]; then
  GPU_CSV=$(capture_bound 256 "$NVIDIA_SMI" \
    --query-gpu=name,driver_version,compute_cap --format=csv,noheader)
  GPU_LINE=$(printf '%s\n' "$GPU_CSV" | head -n 1)
  GPU_NAME=$(csv_field "$GPU_LINE" 1)
  DRIVER_VERSION=$(csv_field "$GPU_LINE" 2)
  COMPUTE_CAP=$(csv_field "$GPU_LINE" 3)
fi

if [ "$SUITE" -eq 1 ]; then
  [ -n "$BIN_DIR" ] || {
    echo "--suite requires --bin-dir" >&2
    exit 1
  }
  run_suite
else
  [ "${#APP[@]}" -gt 0 ] || usage
  run_one "$TOOL" "${APP[@]}" || true
fi

if [ "${#FAILURES[@]}" -gt 0 ]; then
  echo "${#FAILURES[@]} sanitizer job(s) failed" >&2
  exit 1
fi
echo "${RUN_COUNT} sanitizer job(s) clean"
exit 0
