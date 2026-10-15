#!/usr/bin/env bash
cd /mnt/vm_8tb/b70/build/backend-decision; for m in eager graph; do echo "== $m"; ./e4-run.sh $m; done
