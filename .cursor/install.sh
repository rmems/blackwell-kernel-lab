#!/usr/bin/env bash
# Cursor Cloud Agent install script for blackwell-kernel-lab (`install` in .cursor/environment.json).
#
# Cursor runs this from the repository root during every Build, on its default
# Ubuntu base image (CPU only: cloud agents have no GPU), then snapshots the disk.
# It must be idempotent. Shell exports don't survive into agent runs, so the tools
# it installs are exposed through /etc/profile.d and /usr/local/bin.
# See https://cursor.com/docs/cloud-agent/setup
#
# Installs only what this repo's CI and manifests need:
#   - apt: build-essential, cmake, shellcheck
#   - CPU-only configure: cmake -S kernels -B build/kernels-cpu -DBKL_ENABLE_CUDA=OFF
#
# It ends with a dependency fetch/prebuild, not a test run.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

SUDO=""
if [ "$(id -u)" -ne 0 ]; then
  SUDO="sudo"
fi

# Install apt packages that are not already present.
apt_install() {
  local missing=() pkg
  for pkg in "$@"; do
    if ! dpkg-query -W -f='${Status}' "$pkg" 2>/dev/null | grep -q "install ok installed"; then
      missing+=("$pkg")
    fi
  done
  if [ "${#missing[@]}" -gt 0 ]; then
    $SUDO apt-get -o Acquire::Retries=5 update -qq
    $SUDO env DEBIAN_FRONTEND=noninteractive apt-get -o Acquire::Retries=5 install -y --no-install-recommends "${missing[@]}"
  fi
}

# --- System packages (cmake + C++ compiler for the CPU configure path; shellcheck for ci-lint.yml) ---
apt_install build-essential cmake shellcheck curl ca-certificates

# --- Prebuild: CPU-only CMake configure (ci-cpu.yml) ---
# BKL_ENABLE_CUDA=OFF skips every kernel target, so no nvcc is needed.
cmake -S kernels -B build/kernels-cpu -DBKL_ENABLE_CUDA=OFF

echo "Cursor install for blackwell-kernel-lab finished."
