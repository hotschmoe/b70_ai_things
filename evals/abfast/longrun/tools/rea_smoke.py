#!/usr/bin/env python3
"""No-model proof that Pi 1.2.0 -> MCP -> REA -> Ghidra works inside abfast-long:1.

Starts the scripted mock model, runs one Pi round in a container that has a task's stripped binary at /opt/target,
the mock "model" emits open_binary / list_procedures / procedure_pseudo_code tool calls; prints what Pi returned.
  rea_smoke.py [TASK_ID=re-vault] [BINARY=vault]
"""
import json, os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import lr_common as C, run_long as RL, run_arm as RA

tid = sys.argv[1] if len(sys.argv) > 1 else "re-vault"; binary = sys.argv[2] if len(sys.argv) > 2 else "vault"
port = 19000 + os.getpid() % 900
calls = [{"name": "mcp__rea__open_binary", "arguments": {"path": f"/opt/target/{binary}", "provider_id": "ghidra"}},
         {"name": "mcp__rea__list_procedures", "arguments": {}},
         {"name": "mcp__rea__list_strings", "arguments": {}},
         {"name": "mcp__rea__procedure_pseudo_code", "arguments": {"procedure": os.environ.get("SMOKE_PROC", "FUN_00102b30")}}]
env = dict(os.environ, MOCK_CALLS=json.dumps(calls), MOCK_LOG=f"/tmp/rea_smoke_mock_{os.getpid()}.log")
srv = subprocess.Popen([sys.executable, os.path.join(HERE, "mock_openai.py"), str(port)], env=env)
time.sleep(1)
box = C.LBox(f"abfast-reasmoke-{os.getpid()}")
try:
    box.put_tar(C.tar_dir(f"{C.LDATA}/{tid}/target"), "/opt/target", user="0"); box.exec("chmod -R a+rx /opt/target", user="0")
    mj = RA.pi_models_json(f"http://127.0.0.1:{port}/v1", "x", "hotschmoe-dd", 131072, 32768)
    box.exec("mkdir -p $PI_CODING_AGENT_DIR /logs/pi && cat > $PI_CODING_AGENT_DIR/models.json", stdin=json.dumps(mj).encode())
    RL.setup_rea(box)
    rc, o = box.exec("pi --version; pi mcp list 2>&1 | cut -c1-300 | head -5")
    print(o)
    t0 = time.time()
    rc, o = box.exec("cd /work && pi --print --mode json --session-dir /logs/pi --provider harbor-endpoint --model hotschmoe-dd --thinking xhigh 'analyze the binary' </dev/null", timeout=900)
    print("pi rc", rc, "secs", round(time.time() - t0))
    for line in o.splitlines():
        if '"type":"message_update"' in line:
            continue
        try:
            ev = json.loads(line)
        except Exception:
            continue
        if ev.get("type") == "message_end" and ev["message"].get("role") == "toolResult":
            m = ev["message"]; txt = json.dumps(m.get("content"))
            print("TOOL RESULT", m.get("toolName"), "isError=", m.get("isError"), "len", len(txt), "|", txt[:260].replace("\\n", " "))
    log = [json.loads(l) for l in open(f"/tmp/rea_smoke_mock_{os.getpid()}.log")]
    print("first request: tools declared =", log[0]["tools"], "tools_chars =", log[0]["tools_chars"], "skill_in_system =", log[0]["skill_in_system"])
finally:
    box.rm(); srv.kill()
