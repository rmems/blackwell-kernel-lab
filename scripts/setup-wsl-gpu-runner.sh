#!/usr/bin/env bash
#
# Idempotent ShipOfTheseus registration inside WSL2 Ubuntu 24.04.
#
# The GPU workflows run on the labels self-hosted, Linux, X64, CUDA.
# This script registers those labels. It does not change them.
# It never installs a Linux NVIDIA driver. WSL uses the Windows driver
# through /usr/lib/wsl/lib/nvidia-smi.
#
# Usage (as the Ubuntu user, not root):
#   bash scripts/setup-wsl-gpu-runner.sh
#   RUNNER_TOKEN=... bash scripts/setup-wsl-gpu-runner.sh
#
# The registration token is read from RUNNER_TOKEN or from a silent
# prompt. It is not printed and it is not written into this repo.
# Default runner directory: $HOME/actions-runner

set -euo pipefail

RUNNER_VERSION=2.338.0
RUNNER_SHA256=af4b794c1bc41d73d40535e3fe092a39f9679cd8d965954c2aca25a05ca41d32
RUNNER_REPO=rmems/blackwell-kernel-lab
RUNNER_URL=https://github.com/rmems/blackwell-kernel-lab
RUNNER_NAME="${RUNNER_NAME:-ShipOfTheseus}"
RUNNER_LABELS=self-hosted,Linux,X64,CUDA
RUNNER_ROOT="${RUNNER_ROOT:-${HOME}/actions-runner}"
NVSMI=/usr/lib/wsl/lib/nvidia-smi
NVSMI_ON_PATH=/usr/local/bin/nvidia-smi
# Set to 1 by download_runner only after a checksum-verified extraction.
RUNNER_FRESH=0

refuse() {
  echo "error: $*" >&2
  exit 1
}

as_root() {
  if [[ "$(id -u)" -eq 0 ]]; then
    "$@"
  else
    sudo "$@"
  fi
}

require_wsl2_ubuntu() {
  local release os_id os_version
  release=$(tr -d '\0' </proc/sys/kernel/osrelease 2>/dev/null || true)
  case "$release" in
    *WSL2*) ;;
    *)
      refuse "not WSL2 (kernel release '${release:-unknown}'). Run this inside WSL2 Ubuntu 24.04 on the Windows host. Do not install a Linux NVIDIA driver."
      ;;
  esac
  [[ -f /etc/os-release ]] || refuse "missing /etc/os-release. Ubuntu 24.04 is required."
  os_id=$(awk -F= '$1=="ID"{gsub(/"/,"",$2); print $2}' /etc/os-release)
  os_version=$(awk -F= '$1=="VERSION_ID"{gsub(/"/,"",$2); print $2}' /etc/os-release)
  if [[ "$os_id" != "ubuntu" || "$os_version" != "24.04" ]]; then
    refuse "Ubuntu 24.04 is required (found ${os_id:-unknown} ${os_version:-unknown})."
  fi
}

refuse_linux_nvidia_driver() {
  local package
  if [[ ! -r /var/lib/dpkg/status ]]; then
    return 0
  fi
  while read -r package; do
    case "$package" in
      nvidia-driver-* | nvidia-dkms-* | cuda-drivers | cuda-drivers-*)
        refuse "Linux NVIDIA driver package '${package}' is installed. Remove it and use the Windows driver through ${NVSMI}. This script will not install a Linux driver."
        ;;
      *) ;;
    esac
  done < <(dpkg-query -W -f='${Package}\n' 2>/dev/null || true)
}

require_compute_12() {
  local line cap
  [[ -x "$NVSMI" ]] || refuse "${NVSMI} is missing or not executable. Install the Windows NVIDIA driver for the RTX 5080, then restart WSL. Do not install a Linux NVIDIA driver inside WSL."
  "$NVSMI" >/dev/null || refuse "${NVSMI} failed. The Windows driver is not visible in this distro. Do not install a Linux NVIDIA driver inside WSL."
  local found=0
  while IFS= read -r line; do
    cap=${line//[[:space:]]/}
    [[ -n "$cap" ]] || continue
    found=1
    if [[ "$cap" != "12.0" ]]; then
      refuse "compute capability is '${cap}', expected 12.0 (RTX 5080 / sm_120)."
    fi
  done < <("$NVSMI" --query-gpu=compute_cap --format=csv,noheader)
  [[ "$found" -eq 1 ]] || refuse "${NVSMI} returned no compute capability."
}

install_nvidia_smi_wrapper() {
  local tmp
  if [[ -x "$NVSMI_ON_PATH" ]] && "$NVSMI_ON_PATH" --query-gpu=compute_cap --format=csv,noheader >/dev/null 2>&1; then
    return 0
  fi
  if [[ -e "$NVSMI_ON_PATH" ]]; then
    refuse "${NVSMI_ON_PATH} exists but does not work. GPU jobs call nvidia-smi on PATH. Remove that file and re-run. This script will not install a Linux driver."
  fi
  tmp=$(mktemp)
  cat >"$tmp" <<'EOF'
#!/bin/sh
# blackwell-kernel-lab WSL nvidia-smi wrapper. Not a Linux NVIDIA driver.
exec /usr/lib/wsl/lib/nvidia-smi "$@"
EOF
  as_root install -m 755 "$tmp" "$NVSMI_ON_PATH"
  rm -f "$tmp"
}

require_tools() {
  local tool
  for tool in curl tar sha256sum install; do
    command -v "$tool" >/dev/null 2>&1 || refuse "missing required tool: ${tool}"
  done
  if [[ "$(id -u)" -ne 0 ]]; then
    command -v sudo >/dev/null 2>&1 || refuse "missing required tool: sudo"
  fi
}

refuse_repo_root() {
  local script_dir repo_root
  script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  repo_root=$(cd "${script_dir}/.." && pwd)
  if git -C "$repo_root" rev-parse --show-toplevel >/dev/null 2>&1; then
    repo_root=$(git -C "$repo_root" rev-parse --show-toplevel)
  fi
  # Canonicalize so ".", "./actions-runner", or symlinks cannot bypass the guard.
  RUNNER_ROOT=$(realpath -m -- "$RUNNER_ROOT")
  repo_root=$(realpath -m -- "$repo_root")
  case "$RUNNER_ROOT" in
    "$repo_root" | "$repo_root"/*)
      refuse "RUNNER_ROOT (${RUNNER_ROOT}) is inside the git checkout (${repo_root}). Use a directory outside the repo, for example \$HOME/actions-runner."
      ;;
    *) ;;
  esac
}

download_runner() {
  local archive url stamp deps
  stamp="${RUNNER_ROOT}/.bkl-runner-version"
  deps="${RUNNER_ROOT}/.bkl-runner-deps"
  if [[ -x "${RUNNER_ROOT}/config.sh" && -f "$stamp" && -f "$deps" ]] \
    && [[ "$(tr -d '[:space:]' <"$stamp")" == "$RUNNER_VERSION" ]] \
    && [[ "$(tr -d '[:space:]' <"$deps")" == "$RUNNER_VERSION" ]]; then
    echo "actions/runner ${RUNNER_VERSION} already extracted in ${RUNNER_ROOT}"
    return 0
  fi
  url="https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz"
  archive=$(mktemp)
  # shellcheck disable=SC2064 # archive is set; expand it when the trap is installed
  trap "rm -f '$archive'" RETURN
  echo "downloading actions/runner ${RUNNER_VERSION}"
  curl --fail --silent --show-error --location --retry 3 --output "$archive" "$url"
  printf '%s  %s\n' "$RUNNER_SHA256" "$archive" | sha256sum --check --strict -
  mkdir -p "$RUNNER_ROOT"
  tar -xzf "$archive" -C "$RUNNER_ROOT"
  printf '%s\n' "$RUNNER_VERSION" >"${RUNNER_ROOT}/.bkl-runner-version"
  RUNNER_FRESH=1
}

install_runner_libraries() {
  # installdependencies.sh lives in a directory the runner user (and therefore
  # any GPU job) can write. Only run it as root straight after a verified
  # extraction; never re-run a reused copy with sudo.
  # .bkl-runner-deps is separate from the version stamp. A failed install
  # leaves it absent, and the next run re-extracts the verified archive first.
  if [[ "$RUNNER_FRESH" != "1" ]]; then
    echo "skipping installdependencies.sh: runner files were reused, not freshly verified"
    return 0
  fi
  echo "installing actions/runner library dependencies (not an NVIDIA driver)"
  as_root "${RUNNER_ROOT}/bin/installdependencies.sh"
  printf '%s\n' "$RUNNER_VERSION" >"${RUNNER_ROOT}/.bkl-runner-deps"
}

drop_local_registration() {
  # config.sh refuses an already-configured directory even with --replace.
  # Stop and uninstall the unit first. Registration files are removed only
  # after that succeeds, so a live service is not left pointing at deleted files.
  (
    cd "$RUNNER_ROOT"
    if [[ -f .service && -x ./svc.sh ]] && systemd_booted; then
      as_root ./svc.sh stop
      as_root ./svc.sh uninstall
    fi
    rm -f .runner .credentials .credentials_rsaparams .service
  )
}

read_token() {
  if [[ -n "${RUNNER_TOKEN:-}" ]]; then
    return 0
  fi
  if [[ ! -t 0 ]]; then
    refuse "RUNNER_TOKEN is unset and stdin is not a terminal. Export the one-hour registration token in RUNNER_TOKEN. This script does not print it."
  fi
  printf 'GitHub runner registration token (not echoed): ' >&2
  read -r -s RUNNER_TOKEN || true
  printf '\n' >&2
  [[ -n "${RUNNER_TOKEN:-}" ]] || refuse "empty registration token"
}

configure_runner() {
  local token
  local -a args
  read_token
  token=$RUNNER_TOKEN
  unset RUNNER_TOKEN
  args=(
    --unattended
    --url "$RUNNER_URL"
    --token "$token"
    --name "$RUNNER_NAME"
    --labels "$RUNNER_LABELS"
    --work _work
  )
  if [[ "${RUNNER_REPLACE:-}" == "1" ]]; then
    if [[ -f "${RUNNER_ROOT}/.runner" ]]; then
      drop_local_registration
    fi
    args+=(--replace)
  fi
  echo "configuring ${RUNNER_NAME} for ${RUNNER_REPO} with labels ${RUNNER_LABELS}"
  (
    cd "$RUNNER_ROOT"
    ./config.sh "${args[@]}"
  )
  unset token
}

systemd_booted() {
  [[ -d /run/systemd/system ]] && command -v systemctl >/dev/null 2>&1
}

wsl_systemd_enabled() {
  [[ -f /etc/wsl.conf ]] && grep -Eq '^[[:space:]]*systemd[[:space:]]*=[[:space:]]*true[[:space:]]*$' /etc/wsl.conf
}

note_systemd() {
  if wsl_systemd_enabled; then
    return 0
  fi
  cat >&2 <<'EOF'
systemd is not enabled for this distro. Add these lines to /etc/wsl.conf:

[boot]
systemd=true

Then, from Windows PowerShell, run: wsl.exe --shutdown
Open the distro again and re-run this script. Until systemd is up, start
the runner in the foreground from the runner directory: ./run.sh
EOF
}

install_service() {
  if ! systemd_booted; then
    note_systemd
    echo "fallback: cd ${RUNNER_ROOT} && ./run.sh"
    return 0
  fi
  note_systemd
  (
    cd "$RUNNER_ROOT"
    if [[ ! -x ./svc.sh ]]; then
      refuse "config.sh did not create ${RUNNER_ROOT}/svc.sh"
    fi
    if [[ -f .service ]]; then
      echo "systemd unit already installed; starting it"
      as_root ./svc.sh start
    else
      as_root ./svc.sh install "$(id -un)"
      as_root ./svc.sh start
    fi
    as_root ./svc.sh status || true
  )
}

main() {
  if [[ "$(id -u)" -eq 0 && -z "${RUNNER_ALLOW_RUNASROOT:-}" ]]; then
    refuse "run this script as your Ubuntu user, not root. It uses sudo only for the systemd unit and the nvidia-smi wrapper."
  fi
  RUNNER_ROOT=${RUNNER_ROOT%/}
  require_wsl2_ubuntu
  refuse_linux_nvidia_driver
  require_compute_12
  require_tools
  refuse_repo_root
  install_nvidia_smi_wrapper
  download_runner
  install_runner_libraries
  if [[ -f "${RUNNER_ROOT}/.runner" && "${RUNNER_REPLACE:-}" != "1" ]]; then
    echo "runner already configured; leaving the existing registration in place"
  else
    configure_runner
  fi
  install_service
  cat <<EOF
Labels registered for the GPU jobs: ${RUNNER_LABELS}
Runner directory: ${RUNNER_ROOT}
Online check:
  gh api repos/${RUNNER_REPO}/actions/runners --jq '.runners[] | select(.name=="${RUNNER_NAME}") | {status,labels:[.labels[].name]}'
Keep WSL running at Windows logon (Task Scheduler) and the nvcc / g++ /
compute-sanitizer prerequisites: recipes/wsl-gpu-runner.md
EOF
}

main "$@"
