python3 - <<'P'
import xml.etree.ElementTree as ET, datetime as dt, re
ns = {"e": "http://schemas.microsoft.com/win/2004/08/events/event"}
evs = []
for e in ET.parse("/home/pat/logs/System-export.xml").getroot():
    s = e.find("e:System", ns)
    t = dt.datetime.strptime(s.find("e:TimeCreated", ns).get("SystemTime")[:19], "%Y-%m-%dT%H:%M:%S")
    d = {x.get("Name"): x.text for x in e.iter("{%s}Data" % ns["e"])}
    evs.append((t, s.find("e:Provider", ns).get("Name"), int(s.find("e:EventID", ns).text), d))
k41 = [x for x in evs if x[1].endswith("Kernel-Power") and x[2] == 41]
bs = sorted(x for x in k41 if x[3].get("BugcheckCode") != "0")
code = next(re.match(r"(0x[0-9a-f]{8})", x[3]["param1"]).group(1) for x in evs if x[2] == 1001)
drv = {x[3]["param1"] for b in bs for x in evs if x[2] == 4101 and dt.timedelta(0) < b[0] - x[0] < dt.timedelta(minutes=5)}
assert len(drv) == 1
open("/home/pat/logs/answer.txt", "w").write("bluescreens=%d\npower_loss=%d\nfirst_bluescreen_utc=%sZ\nbugcheck_code=%s\nsuspect_driver=%s\n" % (
    len(bs), len(k41) - len(bs), bs[0][0].strftime("%Y-%m-%dT%H:%M:%S"), code, drv.pop()))
P
