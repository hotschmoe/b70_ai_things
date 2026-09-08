#!/usr/bin/env bash
# Install boot configuration only; leave any active serving lifecycle running.
set -euo pipefail
[[ "$EUID" == 0 ]] || { echo 'Run this installer with sudo.' >&2; exit 1; }
repo=/mnt/vm_8tb/github/b70_ai_things
source_unit="$repo/vllm/fp8/hotschmoe-dd.service"
[[ -f "$source_unit" ]] || exit 1
systemd-analyze verify "$source_unit"
backup_dir=$(mktemp -d /etc/systemd/system/hotschmoe-dd-backup.XXXXXXXX)
for unit in b70-daily-driver.service hotschmoe-dd.service; do
  if [[ -e "/etc/systemd/system/$unit" ]]; then
    cp -a "/etc/systemd/system/$unit" "$backup_dir/"
  fi
done
install -m 0644 "$source_unit" /etc/systemd/system/hotschmoe-dd.service
systemctl daemon-reload
systemctl enable hotschmoe-dd.service
# Disable the retired boot recipe only after the new unit is installed/enabled.
systemctl disable b70-daily-driver.service
systemctl is-enabled hotschmoe-dd.service
printf 'Saved previous units in %s\n' "$backup_dir"
echo 'Boot setup complete. No running service or manual serve was interrupted.'
echo 'If hotschmoe-dd.service is already running, activate changes with:'
echo '  sudo systemctl restart hotschmoe-dd.service'
echo 'Otherwise, drain/stop any manual serve and wait for its teardown before:'
echo '  sudo systemctl start hotschmoe-dd.service'
