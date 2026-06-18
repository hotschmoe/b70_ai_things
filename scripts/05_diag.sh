#!/usr/bin/env bash
# Verify downloaded model + full PCIe link chain for the B70 (uplink width matters).
set -uo pipefail
ROOT=/mnt/vm_8tb/b70

echo "===== downloaded model files ====="
find "$ROOT/models" -iname '*.gguf' -printf '%s  %p\n' | awk '{printf "%.2f GB  %s\n", $1/1073741824, $2}'

echo
echo "===== PCIe topology (Battlemage + its bridges) ====="
lspci -tv 2>/dev/null | grep -iE 'e2ff|e2f0|e2f1|e223|battlemage|arc' || lspci -tv | head -40

echo
echo "===== link cap/status per node in the chain ====="
for d in 42:00.0 43:01.0 43:02.0 44:00.0; do
  echo "--- $d ---"
  lspci -s "$d" 2>/dev/null | sed 's/^/    /'
  lspci -vv -s "$d" 2>/dev/null | grep -iE 'LnkCap:|LnkSta:' | sed 's/^[[:space:]]*/    /'
done

echo
echo "===== xe driver view of the link (sysfs) ====="
for f in current_link_speed current_link_width max_link_speed max_link_width; do
  v=$(cat /sys/bus/pci/devices/0000:44:00.0/$f 2>/dev/null)
  echo "  44:00.0 $f = ${v:-n/a}"
done
echo "  (also check upstream bridge 0000:42:00.0)"
for f in current_link_speed current_link_width max_link_speed max_link_width; do
  v=$(cat /sys/bus/pci/devices/0000:42:00.0/$f 2>/dev/null)
  echo "  42:00.0 $f = ${v:-n/a}"
done

echo "===== DONE ====="
