#!/usr/bin/env bash
#
# Wait until GPU 0 has enough free VRAM to run kernel CI on this host.
#
# Used by .github/workflows/ci-gpu.yml. The 2 GiB floor matches
# docs/HOST_BASELINE.md. Occupancy is serialized via that floor (and the
# human checklist in README), not an exclusive GPU lock: a train job
# that still leaves ≥2 GiB free is a process-policy miss, not something
# this helper can mutex without a shared lock across agoge-forger.
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
  # Keep values inside signed 32-bit so $((10#$val)) cannot abort with
  # "value too great for base" on a 20-digit string that still matches [0-9]+.
  if [ "${#val}" -gt 9 ]; then
    echo "$name is too large (max 9 digits), got: $val" >&2
    exit 1
  fi
}

as_base10() {
  printf '%d\n' "$((10#$1))"
}

require_uint need_mib "$need_mib"
require_uint wait_s "$wait_s"
require_uint interval_s "$interval_s"

need_mib=$(as_base10 "$need_mib")
wait_s=$(as_base10 "$wait_s")
interval_s=$(as_base10 "$interval_s")

if [ "$interval_s" -lt 1 ]; then
  echo "interval_s must be >= 1, got: $interval_s" >&2
  exit 1
fi

query_free_mib() {
  local raw
  raw=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits) ||
    return 1
  raw=${raw%%$'\n'*}
  raw=${raw//[[:space:]]/}
  printf '%s\n' "$raw"
}

list_compute_apps() {
  echo "compute apps:"
  nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv ||
    true
}

sleep_remaining() {
  local remaining=$1
  local sleeptime=$interval_s
  if [ "$sleeptime" -gt "$remaining" ]; then
    sleeptime=$remaining
  fi
  if [ "$sleeptime" -lt 1 ]; then
    sleeptime=1
  fi
  sleep "$sleeptime"
}

nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv ||
  true
list_compute_apps

# Count from this process start, not a parent shell that sourced us.
SECONDS=0
deadline=$wait_s
saw_ok_probe=0
last_ok_free=

fail_deadline() {
  if [ "$saw_ok_probe" -eq 1 ]; then
    echo "Need at least ${need_mib} MiB free on GPU 0 after waiting ${wait_s}s; last reading ${last_ok_free} MiB" >&2
  else
    echo "nvidia-smi never returned a parseable free-MiB reading after waiting ${wait_s}s" >&2
  fi
  list_compute_apps
  exit 1
}

while true; do
  if free="$(query_free_mib)"; then
    case "$free" in
      '' | *[!0-9]*)
        echo "could not parse free MiB from nvidia-smi: ${free}" >&2
        list_compute_apps
        ;;
      *)
        saw_ok_probe=1
        last_ok_free=$free
        if [ "$free" -ge "$need_mib" ]; then
          echo "GPU 0 has ${free} MiB free (>= ${need_mib})"
          exit 0
        fi
        echo "GPU busy (${free} MiB free)"
        list_compute_apps
        ;;
    esac
  else
    echo "nvidia-smi probe failed" >&2
    list_compute_apps
  fi

  remaining=$((deadline - SECONDS))
  if [ "$remaining" -le 0 ]; then
    fail_deadline
  fi
  echo "waiting (${remaining}s left before fail)"
  sleep_remaining "$remaining"
done
