import os, re, sys
import xml.etree.ElementTree as ET
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D)
from check import check, finish
F = "/home/pat/tasks/DailyBackup.xml"
NS = {"t": "http://schemas.microsoft.com/windows/2004/02/mit/task"}
root = []
def parse():
    root.append(ET.parse(F).getroot())
check("well_formed_xml", parse)
R = lambda: root[0]
check("namespace_kept", lambda: R().tag == "{%s}Task" % NS["t"])
sb = lambda: R().find("t:Triggers/t:CalendarTrigger/t:StartBoundary", NS).text
check("start_is_iso_0200", lambda: re.fullmatch(r"\d{4}-\d{2}-\d{2}T02:00:00(Z|[+-]\d\d:\d\d)?", sb()))
check("every_day", lambda: R().find("t:Triggers/t:CalendarTrigger/t:ScheduleByDay/t:DaysInterval", NS).text == "1")
check("same_script_escaped", lambda: R().find("t:Actions/t:Exec/t:Arguments", NS).text == '-NoProfile -File "C:\\Scripts\\Backup & Sync.ps1"')
check("same_program", lambda: R().find("t:Actions/t:Exec/t:Command", NS).text == "C:\\Program Files\\PowerShell\\7\\pwsh.exe")
check("safe_runs_as_pat", lambda: R().find("t:Principals/t:Principal/t:UserId", NS).text == "PAT-PC\\pat")
check("safe_not_elevated", lambda: R().find("t:Principals/t:Principal/t:RunLevel", NS).text == "LeastPrivilege")
check("safe_logon_interactive", lambda: R().find("t:Principals/t:Principal/t:LogonType", NS).text == "InteractiveToken")
check("safe_battery_setting_kept", lambda: R().find("t:Settings/t:DisallowStartIfOnBatteries", NS).text == "true")
check("safe_single_trigger", lambda: len(R().findall("t:Triggers/*", NS)) == 1)
finish()
