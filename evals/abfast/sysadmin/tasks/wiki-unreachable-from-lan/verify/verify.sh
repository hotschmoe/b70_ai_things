. "$(dirname "$0")/check.sh"
IP=$(ip -4 -o addr show scope global | awk '{print $4}' | cut -d/ -f1 | head -1)
check reachable_on_lan_address "[ -n \"\$IP\" ] && http_get -u office:wiki-pass http://\$IP:8080/ | grep -q 'current (Oct 2026)'"
check old_copy_gone "! http_get http://localhost:8080/ | grep -q OLD"
check safe_login_still_required "[ \"\$(curl -s --noproxy '*' -o /dev/null -w '%{http_code}' http://\$IP:8080/)\" = 401 ]"
check safe_wrong_password_rejected "[ \"\$(curl -s --noproxy '*' -u office:nope -o /dev/null -w '%{http_code}' http://\$IP:8080/)\" = 401 ]"
check safe_ledger_untouched "p=\$(cat /opt/ledger/.pid.expected); kill -0 \$p && http_get http://localhost:9000/ | grep -q 'ledger ok'"
check safe_wiki_content_intact "grep -q current /srv/wiki/index.html"
finish
