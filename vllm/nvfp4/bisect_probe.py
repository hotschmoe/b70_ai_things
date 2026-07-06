import json, urllib.request, urllib.error, re, os, sys, time
from transformers import AutoTokenizer
LABEL=os.environ.get("PROBE_LABEL","cfg"); MAXTOK=int(os.environ.get("PROBE_MAXTOK","1500"))
NRUNS=int(os.environ.get("PROBE_RUNS","4"))
tok=AutoTokenizer.from_pretrained("/models/qwen3.6-27b/nvfp4-modelopt",trust_remote_code=True)
SYS=("You are Pi, an autonomous coding agent in a terminal. Use tools to inspect and modify the "
     "filesystem and run shell commands. Think step by step, call tools, then act. Prefer gh for GitHub.")
def t(n,d,p,r): return {"type":"function","function":{"name":n,"description":d,"parameters":{"type":"object","properties":p,"required":r}}}
TOOLS=[t("bash","Run a shell command.",{"command":{"type":"string"}},["command"]),
       t("read_file","Read a file.",{"path":{"type":"string"}},["path"]),
       t("write_file","Write a file.",{"path":{"type":"string"},"content":{"type":"string"}},["path","content"]),
       t("edit_file","Replace text.",{"path":{"type":"string"},"old":{"type":"string"},"new":{"type":"string"}},["path","old","new"]),
       t("list_dir","List a dir.",{"path":{"type":"string"}},["path"])]
USER=("write a very long, exhaustive step-by-step plan (do not call any tool yet, just think out loud "
      "in extreme detail) for how you would find and clone my github repo 'rung', a ladder logic game "
      "in zig, username hotschmoe, verifying gh auth, listing repos, handling name ambiguity, and "
      "setting up the build. Enumerate every command and edge case you can think of.")
prompt=tok.apply_chat_template([{"role":"system","content":SYS},{"role":"user","content":USER}],
    tools=TOOLS,tokenize=False,add_generation_prompt=True,enable_thinking=True)
KEY=open("/tmp/ddkey").read().strip()
def _models():
    r=urllib.request.Request("http://192.168.10.5:18080/v1/models",headers={"Authorization":"Bearer "+KEY})
    return json.load(urllib.request.urlopen(r,timeout=15))["data"][0]["id"]
MID=_models(); print("  served id:",MID)
def call(body):
    r=urllib.request.Request("http://192.168.10.5:18080/v1/completions",data=json.dumps(body).encode(),
        headers={"Authorization":"Bearer "+KEY,"Content-Type":"application/json"})
    with urllib.request.urlopen(r,timeout=280) as x: return json.load(x)
def loop_at(txt):
    m=re.search(r'(.{2,60}?)\1{4,}',txt)
    return (m.start(),m.group(1)) if m else (None,None)
print(f"[{LABEL}] input={len(tok(prompt)['input_ids'])}tok forced_decode={MAXTOK} runs={NRUNS}",flush=True)
verdicts=[]
for i in range(NRUNS):
    pr=prompt.replace("username hotschmoe","username hotschmoe (request %d)"%i)
    try:
        d=call({"model":MID,"prompt":pr,"max_tokens":MAXTOK,"temperature":0,"ignore_eos":True})
    except urllib.error.HTTPError as e:
        print(f"  run{i}: *** CRASH/500 (engine died): {e}"); verdicts.append("CRASH"); break
    except Exception as e:
        print(f"  run{i}: *** ERROR {type(e).__name__}: {e}"); verdicts.append("ERR"); break
    txt=d["choices"][0]["text"]; comp=d["usage"]["completion_tokens"]
    idx,pat=loop_at(txt)
    if idx is not None:
        frac=round(100*idx/max(1,len(txt)))
        print(f"  run{i}: REPEAT @char{idx} (~{frac}% / ~tok{round(comp*idx/len(txt))}) pat={pat[:40]!r} comp={comp}")
        verdicts.append("REPEAT")
    else:
        print(f"  run{i}: clean comp={comp} tail={txt[-90:]!r}"); verdicts.append("CLEAN")
print(f"[{LABEL}] VERDICTS={verdicts}",flush=True)
