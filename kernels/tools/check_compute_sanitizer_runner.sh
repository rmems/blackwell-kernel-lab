#!/usr/bin/env bash
#
# CPU tests for kernels/tools/run_compute_sanitizer.sh.
# Creates fake compute-sanitizer / nvcc / nvidia-smi binaries so
# GitHub-hosted CI can prove: clean logs pass, findings fail, ERROR
# SUMMARY is required, logs are truncated, and metadata is recorded.
#
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
RUNNER="${ROOT}/kernels/tools/run_compute_sanitizer.sh"
FIXTURES="${ROOT}/fixtures/compute-sanitizer"

fail() {
  echo "$1" >&2
  exit 1
}

write_exec() {
  local path=$1
  cat > "$path"
  chmod +x "$path"
}

make_fakes() {
  local bindir=$1
  mkdir -p "$bindir"

  write_exec "${bindir}/compute-sanitizer" <<'EOF'
#!/bin/sh
if [ "${1:-}" = "--version" ]; then
  echo "NVIDIA Compute Sanitizer, Version 2025.3.0"
  exit 0
fi
while [ $# -gt 0 ]; do
  case "$1" in
    --tool|--error-exitcode|--print-limit|--check-exit-code)
      shift 2
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "unexpected sanitizer flag: $1" >&2
      exit 99
      ;;
    *)
      break
      ;;
  esac
done
if [ $# -eq 0 ]; then
  echo "fake compute-sanitizer: missing application" >&2
  exit 1
fi
printf '%s' "${FAKE_SANITIZER_PAD:-}"
"$@"
status=$?
errors=${FAKE_SANITIZER_ERRORS:-0}
echo "========= COMPUTE-SANITIZER"
if [ "${FAKE_SANITIZER_OMIT_SUMMARY:-}" = "1" ]; then
  exit "$status"
fi
if [ "$errors" = "1" ]; then
  echo "========= ERROR SUMMARY: 1 error"
else
  echo "========= ERROR SUMMARY: ${errors} errors"
fi
if [ "$errors" != "0" ]; then
  exit "${FAKE_SANITIZER_EXIT:-1}"
fi
exit "$status"
EOF

  write_exec "${bindir}/nvcc" <<'EOF'
#!/bin/sh
echo 'nvcc: NVIDIA (R) Cuda compiler driver'
echo 'Cuda compilation tools, release 13.3, V13.3.73'
EOF

  write_exec "${bindir}/nvidia-smi" <<'EOF'
#!/bin/sh
echo 'NVIDIA GeForce RTX 5080, 610.43.03, 12.0'
EOF

  write_exec "${bindir}/bkl_device_hello" <<'EOF'
#!/bin/sh
echo 'bkl device_hello sm_120 ok'
EOF
  cp "${bindir}/bkl_device_hello" "${bindir}/bkl_graph_launch_bench"
  cp "${bindir}/bkl_device_hello" "${bindir}/bkl_green_ctx_bench"
}

[ -f "${FIXTURES}/memcheck-clean.log" ] || fail "missing clean fixture"
[ -f "${FIXTURES}/memcheck-finding.log" ] || fail "missing finding fixture"
grep -q 'ERROR SUMMARY: 0 errors' "${FIXTURES}/memcheck-clean.log"
grep -q 'ERROR SUMMARY: 1 error' "${FIXTURES}/memcheck-finding.log"
grep -q 'bkl_oob_example' "${FIXTURES}/memcheck-finding.log"

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
bindir="${tmp}/bin"
make_fakes "$bindir"
export PATH="${bindir}:${PATH}"
unset COMPUTE_SANITIZER NVCC NVIDIA_SMI || true

out="${tmp}/clean"
mkdir -p "$out"
set +e
"$RUNNER" --out-dir "$out" -- "${bindir}/bkl_device_hello" \
  >"${tmp}/clean.stdout" 2>"${tmp}/clean.stderr"
status=$?
set -e
[ "$status" -eq 0 ] || fail "clean run failed: $(cat "${tmp}/clean.stderr")"
python3 - "${out}/memcheck-bkl_device_hello.json" <<'PY'
import json
import sys
from pathlib import Path

meta = json.loads(Path(sys.argv[1]).read_text())
assert meta["schema"] == "bkl.compute_sanitizer.v1"
assert meta["error_summary"] == 0
assert meta["exit_code"] == 0
assert meta["tool"] == "memcheck"
assert meta["binary"]["sha256"]
assert "Compute Sanitizer" in (meta["sanitizer_version"] or "")
assert "13.3" in (meta["cuda"]["nvcc_version"] or "")
assert "RTX 5080" in (meta["gpu"]["name"] or "")
assert meta["driver_version"] == "610.43.03"
assert "--error-exitcode" in meta["command"]
assert meta["log_truncated"] is False
PY

out="${tmp}/finding"
mkdir -p "$out"
set +e
FAKE_SANITIZER_ERRORS=1 "$RUNNER" --out-dir "$out" -- "${bindir}/bkl_device_hello" \
  >"${tmp}/finding.stdout" 2>"${tmp}/finding.stderr"
status=$?
set -e
[ "$status" -ne 0 ] || fail "findings must fail the runner"
grep -q 'error_summary=1' "${tmp}/finding.stderr"
python3 - "${out}/memcheck-bkl_device_hello.json" <<'PY'
import json
import sys
from pathlib import Path

meta = json.loads(Path(sys.argv[1]).read_text())
assert meta["error_summary"] == 1
PY

out="${tmp}/nosummary"
mkdir -p "$out"
set +e
FAKE_SANITIZER_OMIT_SUMMARY=1 "$RUNNER" --out-dir "$out" -- "${bindir}/bkl_device_hello" \
  >"${tmp}/nosummary.stdout" 2>"${tmp}/nosummary.stderr"
status=$?
set -e
[ "$status" -ne 0 ] || fail "missing ERROR SUMMARY must fail"
grep -q 'missing ERROR SUMMARY' "${tmp}/nosummary.stderr"

out="${tmp}/trunc"
mkdir -p "$out"
set +e
FAKE_SANITIZER_PAD=$(printf 'x%.0s' {1..400}) "$RUNNER" --out-dir "$out" \
  --max-log-bytes 128 -- "${bindir}/bkl_device_hello" \
  >"${tmp}/trunc.stdout" 2>"${tmp}/trunc.stderr"
status=$?
set -e
[ "$status" -eq 0 ] || fail "truncated clean run failed: $(cat "${tmp}/trunc.stderr")"
python3 - "${out}/memcheck-bkl_device_hello.json" <<'PY'
import json
import sys
from pathlib import Path

meta = json.loads(Path(sys.argv[1]).read_text())
assert meta["log_truncated"] is True
assert meta["log_bytes"] <= 160
assert "[truncated]" in Path(meta["log_path"]).read_text()
PY

out="${tmp}/suite"
mkdir -p "$out"
set +e
"$RUNNER" --suite --bin-dir "$bindir" --out-dir "$out" \
  >"${tmp}/suite.stdout" 2>"${tmp}/suite.stderr"
status=$?
set -e
[ "$status" -eq 0 ] || fail "suite failed: $(cat "${tmp}/suite.stderr")"
python3 - "${out}/summary.json" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text())
assert summary["schema"] == "bkl.compute_sanitizer.suite.v1"
assert len(summary["runs"]) == 5, len(summary["runs"])
tools = {(run["tool"], Path(run["binary"]["path"]).name) for run in summary["runs"]}
assert ("memcheck", "bkl_device_hello") in tools
assert ("initcheck", "bkl_device_hello") in tools
assert ("memcheck", "bkl_graph_launch_bench") in tools
assert ("initcheck", "bkl_graph_launch_bench") in tools
assert ("memcheck", "bkl_green_ctx_bench") in tools
PY

echo "compute sanitizer runner: ok"
