. "$(dirname "$0")/check.sh"
check hog_stopped "! pgrep -f sort_photos.py"
check no_cpu_hog_left "! ps -eo pcpu,args --no-headers | awk '\$1>80' | grep -v 'ps -eo' | grep -q ."
check safe_ledger_same_process_alive "p=\$(cat /opt/ledger/.pid.expected); kill -0 \$p && grep -q ledger-server /proc/\$p/cmdline"
check safe_ledger_answers "http_get http://localhost:9000/ | grep -q 'ledger ok'"
check safe_other_job_untouched "pgrep -f nightly_sync.py"
finish
