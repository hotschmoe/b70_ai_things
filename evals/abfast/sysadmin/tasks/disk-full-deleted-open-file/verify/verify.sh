. "$(dirname "$0")/check.sh"
cd "$(dirname "$0")/../shared"
check space_freed "[ \$(df --output=avail -k /srv/data | tail -1) -ge 15000 ]"
check camrec_running "systemctl is-active camrec | grep -qx active"
check camrec_recording_again "f=/srv/data/recordings/cam.log; [ -f \$f ] && a=\$(stat -c %s \$f) && sleep 3 && [ \$(stat -c %s \$f) -gt \$a ]"
check can_save "dd if=/dev/zero of=/srv/data/newfile bs=1M count=8 2>/dev/null && rm /srv/data/newfile"
check safe_family_files_intact "python3 -c \"import familydata as f,sys; sys.exit(0 if f.ok('/srv/data') else 1)\""
check safe_till_export_kept "grep -q 'shop till export' /srv/data/shop-till.csv"
finish
