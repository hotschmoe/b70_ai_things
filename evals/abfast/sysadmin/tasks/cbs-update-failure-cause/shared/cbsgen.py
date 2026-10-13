import random, datetime as dt
ANSWER = {"failed_update": "KB5099911", "error_code": "0x80070070", "root_cause": "disk_full", "session_id": "30551234_7788991"}
CODES = {"0x800f0922": "pending_reboot", "0x80073712": "corrupt_component_store", "0x80070002": "missing_file",
         "0x80070005": "access_denied", "0x80070070": "disk_full"}
NAMES = {"0x800f0922": "CBS_E_INSTALLERS_FAILED", "0x80073712": "ERROR_SXS_COMPONENT_STORE_CORRUPT", "0x80070002": "ERROR_FILE_NOT_FOUND",
         "0x80070005": "E_ACCESSDENIED", "0x80070070": "ERROR_DISK_FULL"}
def L(t, lvl, msg): return f"{t.strftime('%Y-%m-%d %H:%M:%S')}, {lvl:<20}CBS    {msg}"
def noise(r, t, pkg):
    kinds = ["Appl: detect Parent, Package: {p}, Parent: Package_for_RollupFix~31bf3856ad364e35~amd64~~22621.2428.1.9, Disposition = Detect, VersionComp: Newer",
             "Exec: Processing package: {p}", "Read out cached package applicability for package: {p}, ApplicableState: 112, CurrentState:112",
             "Loaded servicing stack 10.0.22621.2506 with Core: C:\\Windows\\WinSxS\\amd64_microsoft-windows-servicingstack_31bf3856ad364e35_10.0.22621.2506_none\\cbscore.dll",
             "Warning: Unrecognized packageExtended attribute.", "Plan: Package: {p}, Identity: Package_for_X~31bf3856ad364e35~amd64~~1.0.0.0, state: Installed",
             "Ending TrustedInstaller finalization.", "Idle processing thread terminated normally", "Starting TrustedInstaller initialization.",
             "Failed to get next phase for IMsiFinalizerState hr=0x80004005 [E_FAIL] (ignored: no MSI installed)"]
    return L(t, r.choice(["Info", "Info", "Info", "Warning"]), r.choice(kinds).format(p=pkg))
def session(r, t, sid, kb, outcome):
    """outcome: ('ok',) or ('fail', code, retried_ok)"""
    pkg = f"Package_for_{kb}~31bf3856ad364e35~amd64~~10.0.1.{r.randrange(9)}"
    lines = [L(t, "Info", f"Session: {sid} initialized by client WindowsUpdateAgent.")]
    for i in range(r.randrange(25, 45)):
        t += dt.timedelta(seconds=r.randrange(1, 4)); lines.append(noise(r, t, pkg))
    lines.append(L(t, "Info", f"Exec: Processing package: {pkg} (installing {kb})"))
    # benign recoverable noise
    lines.append(L(t, "Warning", "Failed to open file C:\\Windows\\WinSxS\\Temp\\InFlight\\old.manifest hr=0x80070002 [ERROR_FILE_NOT_FOUND], retrying - succeeded"))
    if outcome[0] == "ok":
        lines.append(L(t, "Info", f"Session: {sid} - package {kb} installed, state: Installed, hr=0x00000000"))
    else:
        code = outcome[1]
        t += dt.timedelta(seconds=2)
        lines.append(L(t, "Error", f"Failed to perform operation on {pkg}, hr={code} [{NAMES[code]}]"))
        lines.append(L(t, "Error", f"Failed execution of queue item Installer: {pkg} hr={code}"))
        lines.append(L(t, "Info", f"Session: {sid} - package {kb} rolled back, state: Absent, hr={code}"))
        lines.append(L(t, "Error", f"Session: {sid} finished with failure"))
    return t, lines
def build():
    r = random.Random(5); t = dt.datetime(2026, 8, 20, 3, 0, 0); out = []
    plan = [("KB5099800", ("ok",)), ("KB5099804", ("fail", "0x80070002")), ("KB5099804", ("ok",)), ("KB5099805", ("ok",)),
            ("KB5099890", ("fail", "0x80073712")), ("KB5099890", ("ok",)), ("KB5099901", ("ok",)),
            ("KB5099905", ("fail", "0x80070005")), ("KB5099905", ("ok",)), ("KB5099907", ("ok",)),
            ("KB5099911", ("fail", "0x800f0922")),     # first attempt of the October update: pending reboot, user rebooted
            ("KB5099909", ("ok",)), ("KB5099912", ("ok",)),
            ("KB5099911", ("fail", "0x80070070"))]     # MOST RECENT attempt of the October update
    n = len(plan)
    for i in range(30):                                     # filler sessions between
        plan.insert(r.randrange(0, n), (f"KB50998{r.randrange(10, 99)}", ("ok",)))
    # keep the order of the interesting ones: stable sort by planned position is by insertion; fix the last two at the end
    plan = [p for p in plan if p not in (("KB5099911", ("fail", "0x80070070")),)] + [("KB5099911", ("fail", "0x80070070"))]
    for idx, (kb, oc) in enumerate(plan):
        t += dt.timedelta(hours=r.randrange(3, 40))
        sid = "30551234_7788991" if (kb, oc) == ("KB5099911", ("fail", "0x80070070")) else f"{30500000 + r.randrange(99999):08d}_{r.randrange(1000000, 9999999)}"
        t, lines = session(r, t, sid, kb, oc); out += lines
    out.append(L(t, "Info", "Ending TrustedInstaller finalization."))
    return "\n".join(out) + "\n"
