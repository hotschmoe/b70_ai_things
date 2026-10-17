#!/bin/bash
export ABFAST_PI_DIR=/mnt/vm_8tb/b70/tooling/pi-1.3.0
cd /mnt/vm_8tb/github/b70_ai_things
C="--base-url http://192.168.10.5:18080/v1 --api-key-file /mnt/vm_8tb/b70/secrets/dd_api_key --model hotschmoe-dd --parallel 1 --thinking xhigh"
python3 evals/abfast/run_arm.py --tier sysadmin $C --timeout-s 3600 --arm-label sysC2-dd-nightv3-pi130-p1 --tasks $(cat /mnt/vm_8tb/b70/evals/abfast/sys_rest.txt) > sysC2.out 2>&1
python3 evals/abfast/run_arm.py --tier langs $C --timeout-s 3600 --arm-label langsC-dd-nightv3-pi130-p1 --tasks yaml-edit-flow-trailing-comma,gojq-match-empty-after-nonempty,hyperfine-shell-operator-guard,fd-sanitize-error-messages > langsC.out 2>&1
