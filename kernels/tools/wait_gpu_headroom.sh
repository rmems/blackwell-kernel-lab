#!/usr/bin/env bash
#
# Wait until GPU 0 has enough free VRAM to run kernel CI on this host.
#
# Used by .github/workflows/ci-gpu.yml. The 2 GiB floor matches
# docs/HOST_BASELINE.md. Occupancy is serialized: this never starts kernel
# smoke while another compute process holds the card below the gate.
#
# Usage:
#   kernels/tools/wait_gpu_headroom.sh [need_mib] [wait_s] [interval_s]
#
# Defaults: 2048 MiB, 600 s wait, 30 s poll.

set -euo pipefail

need_mib=${1:-2048}
wait_s=${2:-600}
interval_s=${3:-30}

command -v nvidia-smi >/dev/null || {
  echo "missing required tool: nvidia-smi" >&2
  exit 1
}

require_uint() {
  local name=$1
  local val=$2
  case "$val" in
    '' | *[!0-9]*)
      echo "$name must be a non-negative integer, got: $val" >&2
      exit 1
      ;;
  esac
}

require_uint need_mib "$need_mib"
require_uint wait_s "$wait_s"
require_uint interval_s "$interval_s"

if [ "$interval_s" -lt 1 ]; then
  echo "interval_s must be >= 1, got: $interval_s" >&2
  exit 1
fi

free_mib() {
  local raw
  raw=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
  raw=${raw%%$'\n'*}
  raw=${raw//[[:space:]]/}
  printf '%s\n' "$raw"
}

nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv
echo "compute apps:"
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv ||
  true

deadline=$((SECONDS + wait_s))
while true; do
  free="$(free_mib)"
  case "$free" in
    '' | *[!0-9]*)
      echo "could not parse free MiB from nvidia-smi: ${free}" >&2
      exit 1
      ;;
  esac
  if [ "$free" -ge "$need_mib" ]; then
    echo "GPU 0 has ${free} MiB free (>= ${need_mib})"
    exit 0
  fi
  remaining=$((deadline - SECONDS))
  if [ "$remaining" -le 0 ]; then
    echo "Need at least ${need_mib} MiB free on GPU 0; found ${free} MiB after waiting ${wait_s}s" >&2
    exit 1
  fi
  echo "GPU busy (${free} MiB free); waiting ${interval_s}s (${remaining}s left before fail)"
  sleep "$interval_s"
done
