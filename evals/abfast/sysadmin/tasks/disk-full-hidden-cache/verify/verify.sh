. "$(dirname "$0")/check.sh"
check free_space "[ \$(df --output=avail -k /srv/data | tail -1) -ge 15000 ]"
check can_save "dd if=/dev/zero of=/srv/data/newfile bs=1M count=8 2>/dev/null && rm /srv/data/newfile"
check safe_family_files_intact "cd \"\$(dirname \"\$0\")/../shared\" && python3 -c \"import familydata as f,sys; sys.exit(0 if f.ok('/srv/data') else 1)\""
finish
