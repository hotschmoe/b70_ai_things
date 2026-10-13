python3 - <<'P'
import re
lines = open("/home/pat/logs/CBS.log").read().splitlines()
last = [i for i, l in enumerate(lines) if "finished with failure" in l][-1]
sid = re.search(r"Session: (\S+) finished", lines[last]).group(1)
start = next(i for i, l in enumerate(lines) if f"Session: {sid} initialized" in l)
blk = lines[start:last + 1]
kb = re.search(r"package (KB\d+) rolled back", "\n".join(blk)).group(1)
hr = re.search(r"Failed to perform operation.*hr=(0x[0-9a-f]{8}) \[(\w+)\]", "\n".join(blk))
cause = {"ERROR_DISK_FULL": "disk_full", "ERROR_SXS_COMPONENT_STORE_CORRUPT": "corrupt_component_store", "ERROR_FILE_NOT_FOUND": "missing_file",
         "E_ACCESSDENIED": "access_denied", "CBS_E_INSTALLERS_FAILED": "pending_reboot"}[hr.group(2)]
open("/home/pat/logs/answer.txt", "w").write(f"failed_update={kb}\nerror_code={hr.group(1)}\nroot_cause={cause}\nsession_id={sid}\n")
P
