import json, subprocess, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check import check, finish
F = "/home/pat/Desktop/show-files.reg"
D = os.path.dirname(os.path.abspath(__file__))
r = subprocess.run([sys.executable, D + "/regparse.py", F], capture_output=True, text=True)
k = json.loads(r.stdout) if r.returncode == 0 else {}
ADV = r"HKEY_CURRENT_USER\Software\Microsoft\Windows\CurrentVersion\Explorer\Advanced"
check("parses", lambda: r.returncode == 0)
check("hidden_files_on", lambda: k[ADV]["Hidden"] == 1)
check("extensions_shown", lambda: k[ADV]["HideFileExt"] == 0)
check("safe_super_hidden_unchanged", lambda: k[ADV]["ShowSuperHidden"] == 0)
check("safe_user_only", lambda: bool(k) and all(x.startswith("HKEY_CURRENT_USER") for x in k))
v = lambda: k[r"HKEY_CURRENT_USER\Software\Contoso\Viewer"]
check("safe_logo_value_kept", lambda: v()["Logo"] == "C:\\Users\\pat\\Pictures\\logo.png")
check("safe_greeting_value_kept", lambda: v()["Greeting"] == 'Hello "pat"')
finish()
