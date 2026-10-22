# Recipe: ShipOfTheseus GPU runner in WSL2 Ubuntu 24.04

Register the self-hosted runner on the Windows PC that has the RTX 5080,
inside WSL2 Ubuntu 24.04. The GPU jobs keep the labels they already trust:

`self-hosted`, `Linux`, `X64`, `CUDA`

Those labels are set by [`scripts/setup-wsl-gpu-runner.sh`](../scripts/setup-wsl-gpu-runner.sh).
The script does not edit the workflows or `tools/gpu_ci_gate.py`.

The Windows NVIDIA driver is the only GPU driver. The script checks
`/usr/lib/wsl/lib/nvidia-smi` and refuses unless compute capability is
**12.0**. It does not install a Linux NVIDIA driver, and it refuses if a
`nvidia-driver-*`, `nvidia-dkms-*`, or `cuda-drivers` package is already
installed.

## What the workflows still require

Labels are not the whole job. These stay as written; do not relax them
when something is missing inside WSL.

| Check | Where | WSL note |
|---|---|---|
| `runs-on: [self-hosted, Linux, X64, CUDA]` | `ci-gpu.yml` `gpu-kernels`, `ci-gpu-sanitizer.yml` `gpu-sanitizer` | Satisfied by the labels above. `Linux` is a label, not a second OS. |
| `wait_gpu_headroom.sh 2048 600 30` | both GPU jobs | `nvidia-smi` must be on `PATH` and free VRAM must reach **2048 MiB**. The script installs a wrapper at `/usr/local/bin/nvidia-smi` that execs the WSL binary. |
| `nvcc` and host `g++` 6–15 | `ci-gpu.yml` kernel smoke | The Windows driver does not ship `nvcc`. Install the CUDA toolkit **without** a Linux driver package, and `build-essential` so `/usr/bin/g++-13` exists (Ubuntu 24.04). CMake fails closed until both are present. |
| `device0: NVIDIA GeForce RTX 5080 compute 12.0` and `bkl device_hello sm_120 ok` | `ci-gpu.yml` | Printed by the built binaries, not by `nvidia-smi`. |
| `compute-sanitizer` and `nvidia-smi --query-gpu=...` | `ci-gpu-sanitizer.yml` only | Opt-in (manual dispatch or a trusted `main` push that touches kernel sources). Not a pull-request job. `compute-sanitizer` comes with the CUDA toolkit. |

Hosted `GPU validation` / gate jobs stay on `ubuntu-latest`.

## Ubuntu (inside the distro)

```bash
# Confirm the distro sees the Windows GPU. Stop if this is not 12.0.
/usr/lib/wsl/lib/nvidia-smi --query-gpu=name,compute_cap,memory.free --format=csv

# Toolkit and compiler. Do not install nvidia-driver or cuda-drivers.
sudo apt-get update
sudo apt-get install -y build-essential
# Then the WSL CUDA toolkit from NVIDIA's Ubuntu 24.04 repo, toolkit only.

# From a checkout of this repo, as your user (not root):
bash scripts/setup-wsl-gpu-runner.sh
```

The script prompts for the registration token with echo off. To pass it
without putting it in the shell history, export `RUNNER_TOKEN` for that
process only. The script unsets it after `config.sh` and never writes it
into the repo. The runner files live in `$HOME/actions-runner` unless
`RUNNER_ROOT` points somewhere else **outside** the git checkout.

Re-running the script is safe. It skips download when both
`.bkl-runner-version` and `.bkl-runner-deps` match this release. A failed
`installdependencies.sh` leaves the deps stamp absent, so the next run
verifies the archive again, extracts it, and retries that install.
Registration is skipped when `.runner` already exists. Set `RUNNER_REPLACE=1`
plus a new token to replace it. Replacement stops if `svc.sh stop` or
`svc.sh uninstall` fails, and it does not delete the local registration
files until both succeed. `RUNNER_NAME` defaults to `ShipOfTheseus`.

### systemd

`/etc/wsl.conf` needs:

```ini
[boot]
systemd=true
```

If that line is missing, the script says so and does not pretend the unit
installed. After editing the file, shut the distro down from Windows
(command below) and open it again. The script then runs:

```bash
cd "$HOME/actions-runner"
sudo ./svc.sh install "$USER"
sudo ./svc.sh start
sudo ./svc.sh status
```

`config.sh` creates `svc.sh`. If systemd is not running, start the runner
in the foreground instead:

```bash
cd "$HOME/actions-runner"
./run.sh
```

## Windows PowerShell

```powershell
wsl.exe -l -v
# Ubuntu-24.04 should show VERSION 2.

# After changing /etc/wsl.conf:
wsl.exe --shutdown

# Keep the distro alive at logon. WSL stops when nothing is running in it,
# which takes the runner offline. Task Scheduler, At log on:
$action = New-ScheduledTaskAction -Execute "wsl.exe" -Argument "-d Ubuntu-24.04 --exec /usr/bin/sleep infinity"
$trigger = New-ScheduledTaskTrigger -AtLogOn
Register-ScheduledTask -TaskName "WSL Ubuntu-24.04 stays up" -Action $action -Trigger $trigger -Description "Keep the ShipOfTheseus WSL distro running so the GPU runner stays online"
```

`sleep infinity` holds the distro open. With `[boot] systemd=true`, the
runner unit starts at boot and does not depend on that sleep for the
process itself. Without the scheduled task, an idle distro still exits
and the runner disappears until the next `wsl` launch.

## Online check

From Ubuntu, after the service is started:

```bash
systemctl status 'actions.runner.*' --no-pager
gh api repos/rmems/blackwell-kernel-lab/actions/runners --jq '.runners[] | select(.name=="ShipOfTheseus") | {status,labels:[.labels[].name]}'
```

`status` should be `online`. `labels` should include `self-hosted`,
`Linux`, `X64`, and `CUDA`. The same query works in PowerShell when `gh`
is logged in on Windows. The runner is offline until this distro is up
and `svc.sh` (or `run.sh`) is running. Pause it before a training job
that needs the card; GPU CI and a train share the 2048 MiB floor.
