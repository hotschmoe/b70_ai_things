python3 - <<'P'
p = "/home/pat/tasks/DailyBackup.xml"; s = open(p).read()
s = s.replace("2026-10-14 02:00:00", "2026-10-14T02:00:00").replace("<DaysInterval>0<", "<DaysInterval>1<").replace("Backup & Sync", "Backup &amp; Sync")
open(p, "w").write(s)
P
