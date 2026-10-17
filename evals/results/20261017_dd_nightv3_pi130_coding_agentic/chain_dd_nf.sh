#!/bin/bash
export ABFAST_PI_DIR=/mnt/vm_8tb/b70/tooling/pi-1.3.0
cd /mnt/vm_8tb/github/b70_ai_things
C="--base-url http://192.168.10.5:18080/v1 --api-key-file /mnt/vm_8tb/b70/secrets/dd_api_key --model hotschmoe-dd --parallel 2 --thinking xhigh"
python3 evals/abfast/run_arm.py --tier sysadmin $C --arm-label sysC-dd-nightv3-pi130 > /mnt/vm_8tb/b70/evals/abfast/sysC.out 2>&1
python3 evals/abfast/run_arm.py --tier langs $C --timeout-s 2700 --arm-label langsC-dd-nightv3-pi130 --tasks yaml-edit-flow-trailing-comma,markdown-linear-link-definitions,gojq-match-empty-after-nonempty,gotoml-omit-empty-super-tables,hyperfine-shell-operator-guard,bat-number-nonblank,regex-lazy-macro,fd-sanitize-error-messages,json-rpc-method-request-id,ripgrep-git-config-env-vars > /mnt/vm_8tb/b70/evals/abfast/langsC.out 2>&1
