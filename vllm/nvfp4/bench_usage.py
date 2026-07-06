import json, urllib.request, time, sys
EP="http://192.168.10.5:18080"; KEY="sk-b70-a9ea4790de56bf9a693a9e618c4c905f49464566765bed8d"
LABEL=sys.argv[1] if len(sys.argv)>1 else "cfg"
def MID():
    r=urllib.request.Request(EP+"/v1/models",headers={"Authorization":"Bearer "+KEY})
    return json.load(urllib.request.urlopen(r,timeout=15))["data"][0]["id"]
mid=MID()
def post(path,body,stream=False,timeout=180):
    r=urllib.request.Request(EP+path,data=json.dumps(body).encode(),
        headers={"Authorization":"Bearer "+KEY,"Content-Type":"application/json"})
    return urllib.request.urlopen(r,timeout=timeout)
def decode_ts(prompt,n=400):
    t0=time.time(); x=post("/v1/completions",{"model":mid,"prompt":prompt,"max_tokens":n,"temperature":0,"ignore_eos":True})
    d=json.load(x); el=time.time()-t0; c=d["usage"]["completion_tokens"]
    return c/el, c, el
def ttft(prompt):
    t0=time.time()
    x=post("/v1/completions",{"model":mid,"prompt":prompt,"max_tokens":16,"temperature":0,"stream":True})
    for line in x:
        s=line.decode("utf-8","ignore").strip()
        if s.startswith("data:") and s!="data: [DONE]":
            try:
                j=json.loads(s[5:])
                if j["choices"][0].get("text"): return time.time()-t0
            except: pass
    return time.time()-t0
def prefill_pp(prompt):
    t0=time.time(); x=post("/v1/completions",{"model":mid,"prompt":prompt,"max_tokens":1,"temperature":0})
    d=json.load(x); el=time.time()-t0; pt=d["usage"]["prompt_tokens"]
    return pt/el, pt, el
import os
NONCE=str(os.getpid())
short="<|im_start|>user\n"+NONCE+" Hello.<|im_end|>\n<|im_start|>assistant\n"
codep=("<|im_start|>user\n"+NONCE+" Write a Python function that merges two sorted linked lists, with a docstring "
       "and type hints. Then explain the time complexity.<|im_end|>\n<|im_start|>assistant\n")
longp="<|im_start|>user\n"+(NONCE+" The quick brown fox jumps over the lazy dog. "*1400)+"\nSummarize.<|im_end|>\n<|im_start|>assistant\n"
print(f"### {LABEL}  served={mid}",flush=True)
# warmup
decode_ts(short,32)
dts,c,el=decode_ts(short,400); print(f"decode_tg (generic, 400tok): {dts:.1f} t/s   ({c} in {el:.1f}s)",flush=True)
cts,c,el=decode_ts(codep,400);  print(f"decode_tg (coding,  400tok): {cts:.1f} t/s   ({c} in {el:.1f}s)",flush=True)
tf=ttft(short); print(f"TTFT (short prompt):        {tf*1000:.0f} ms",flush=True)
pp,pt,el=prefill_pp(longp); print(f"prefill PP ({pt} tok):     {pp:.0f} tok/s  (TTFT {el:.2f}s)",flush=True)
