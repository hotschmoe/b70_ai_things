#!/bin/bash
export ABFAST_PI_DIR=/mnt/vm_8tb/b70/tooling/pi-1.3.0
cd /mnt/vm_8tb/github/b70_ai_things
python3 evals/abfast/run_arm.py --tier langs --base-url http://192.168.10.5:18080/v1 --api-key-file /mnt/vm_8tb/b70/secrets/dd_api_key --model hotschmoe-dd --parallel 1 --thinking xhigh --timeout-s 3600 --arm-label langsD-dd-nightv3-pi130-p1 --tasks gotoml-omit-empty-super-tables,markdown-linear-link-definitions > /mnt/vm_8tb/b70/evals/abfast/langsD.out 2>&1
