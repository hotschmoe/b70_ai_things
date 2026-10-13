import random, datetime as dt
NS = "http://schemas.microsoft.com/win/2004/08/events/event"
ANSWER = {"bluescreens": "2", "power_loss": "1", "first_bluescreen_utc": "2026-09-10T22:41:07Z",
          "bugcheck_code": "0x0000009f", "suspect_driver": "nvlddmkm"}
def ev(prov, eid, t, data=None, level=4):
    d = "".join(f'<Data Name="{k}">{v}</Data>' for k, v in (data or {}).items())
    return (t, f'<Event xmlns="{NS}"><System><Provider Name="{prov}"/><EventID>{eid}</EventID><Level>{level}</Level>'
               f'<TimeCreated SystemTime="{t.strftime("%Y-%m-%dT%H:%M:%S")}.{t.microsecond:06d}0Z"/><Computer>PAT-PC</Computer></System>'
               f'<EventData>{d}</EventData></Event>')
def build():
    r = random.Random(11); base = dt.datetime(2026, 9, 7, 0, 0, 0); out = []
    noise = [("Service Control Manager", 7036, {"param1": "Windows Update", "param2": "running"}),
             ("Service Control Manager", 7036, {"param1": "Print Spooler", "param2": "stopped"}),
             ("Microsoft-Windows-DistributedCOM", 10016, {"param1": "machine-default", "param2": "Local Activation"}),
             ("Microsoft-Windows-DNS-Client", 1014, {"param1": "wpad"}),
             ("disk", 153, {"param1": "\\Device\\Harddisk0\\DR0"}),
             ("Microsoft-Windows-Kernel-General", 12, {"param1": "10.0.22631"}),
             ("Microsoft-Windows-Time-Service", 134, {"param1": "time.windows.com"}),
             ("Microsoft-Windows-WindowsUpdateClient", 19, {"param1": "Security Update"})]
    for _ in range(760):
        p, e, d = r.choice(noise)
        out.append(ev(p, e, base + dt.timedelta(seconds=r.randrange(8 * 86400), microseconds=r.randrange(10**6)), d, level=r.choice([2, 3, 4])))
    # decoy GPU hiccups (recovered) with the integrated driver, one shortly before the power-loss restart
    for day, hh, mm in [(8, 10, 5), (11, 14, 30), (12, 6, 49), (13, 9, 2)]:
        out.append(ev("Display", 4101, dt.datetime(2026, 9, day, hh, mm, 12), {"param1": "igdkmd64"}, 3))
    # blue screen restarts: the NVIDIA driver stopped responding right before each
    for t0, code in [(dt.datetime(2026, 9, 10, 22, 41, 7, 311000), 159), (dt.datetime(2026, 9, 14, 3, 12, 8, 120000), 159)]:
        out.append(ev("Display", 4101, t0 - dt.timedelta(minutes=2, seconds=31), {"param1": "nvlddmkm"}, 3))
        out.append(ev("Display", 4101, t0 - dt.timedelta(minutes=1, seconds=10), {"param1": "nvlddmkm"}, 3))
        out.append(ev("Microsoft-Windows-Kernel-Power", 41, t0, {"BugcheckCode": code, "PowerButtonTimestamp": 0}, 1))
        out.append(ev("Microsoft-Windows-WER-SystemErrorReporting", 1001, t0 + dt.timedelta(seconds=9),
                      {"param1": "0x0000009f (0x0000000000000003, 0xffffd30a, 0xffffd30b, 0xffffd30c)"}, 2))
    # power loss restart (no bugcheck)
    out.append(ev("Microsoft-Windows-Kernel-Power", 41, dt.datetime(2026, 9, 12, 7, 5, 44, 905000), {"BugcheckCode": 0, "PowerButtonTimestamp": 0}, 1))
    out.append(ev("EventLog", 6008, dt.datetime(2026, 9, 12, 7, 5, 51), {"param1": "07:02:10", "param2": "9/12/2026"}, 2))
    out.sort(key=lambda x: x[0])
    return '<?xml version="1.0" encoding="utf-8"?>\n<Events>\n' + "\n".join(x[1] for x in out) + "\n</Events>\n"
