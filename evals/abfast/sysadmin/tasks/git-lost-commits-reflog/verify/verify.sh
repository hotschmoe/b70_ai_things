. "$(dirname "$0")/check.sh"
cd /home/maria/website
check on_main "[ \"\$(git symbolic-ref --short HEAD)\" = main ]"
check work_back "for s in 'Add contact page' 'Fix footer' 'Update prices'; do git log main --format=%s | grep -qx \"\$s\" || exit 1; done; [ -f contact.html ] && [ -f footer.html ] && grep -q 'cake 20' prices.txt"
check typo_fix_kept "git log main --format=%s | grep -qx 'Fix typo in about' && grep -qx 'Home page' about.txt"
check tree_clean "[ -z \"\$(git status --porcelain)\" ]"
check no_markers "! grep -rq '<<<<<<<' --exclude-dir=.git ."
check safe_staging_untouched "[ \"\$(git log staging --format=%s | head -1)\" = 'Staging banner' ] && [ \$(git rev-list --count staging) -eq 3 ]"
check safe_original_commit_ids_kept "[ \$(git rev-list --count main) -ge 6 ]"
finish
