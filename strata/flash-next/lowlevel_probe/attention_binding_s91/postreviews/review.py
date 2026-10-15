"""ROOT invokes once/model after frozen S90 packet and full source read."""
from pathlib import Path
import hashlib,json,os,re,subprocess,sys,time
P=Path(__file__).resolve().parent
allowed=['claude-opus-5-5','claude-fable-5-1']
if len(sys.argv)!=4 or sys.argv[1] not in allowed:
    raise SystemExit('usage: review.py MODEL ROOT_FROZEN_PACKET EXPECTED_SHA256; root GO required')
model=sys.argv[1];packet=Path(sys.argv[2]).resolve();expected=sys.argv[3]
if not re.fullmatch('[0-9a-f]{64}',expected):raise SystemExit('exact frozen packet SHA256 required')
sha=lambda b:hashlib.sha256(b).hexdigest()
if not packet.is_file():raise SystemExit('root frozen packet missing; no invocation')
packet_bytes=packet.read_bytes()
if sha(packet_bytes)!=expected:raise SystemExit('root frozen packet SHA mismatch; no invocation')
body=packet_bytes.decode('ascii')
if not body.strip():raise SystemExit('nonempty ASCII root packet required')
cli=Path('/home/hotschmoe/.local/bin/claude')
if not cli.is_file() or not os.access(cli,os.X_OK):raise SystemExit('actual Claude CLI unavailable; no invocation')
# New empty cwd + no resume/continue/session-history options: fresh independent context.
d=P/model;d.mkdir(exist_ok=False)
prompt=(P/'review-prefix.txt').read_text(encoding='ascii')+'\nROOT FROZEN PACKET:\n'+body
cmd=[str(cli),'-p','--model',model,'--effort','low','--tools','','--strict-mcp-config',
     '--disable-slash-commands','--setting-sources','','--settings','{"disableAllHooks":true}',
     '--permission-mode','dontAsk','--max-turns','1','--output-format','stream-json','--verbose']
assert not any(x in cmd for x in ['--resume','--continue','--fork-session'])
metadata={'argv':cmd,'cwd':str(d),'fresh_session_requested':True,'source_packet':str(packet),
          'source_packet_sha256':expected,'prompt_sha256':sha(prompt.encode('ascii')),
          'scope':'S91 CPU-only binding/resource disposition and S92 corrected-value source review; no tool/build/GPU execution; no other model answer'}
with(d/'command.json').open('x',encoding='ascii') as f:json.dump(metadata,f,indent=2);f.write('\n')
with(d/'prompt.txt').open('x',encoding='ascii') as f:f.write(prompt)
t=time.monotonic();exception=None;rc=None
try:
    with(d/'events.jsonl').open('xb') as f,(d/'stderr.log').open('xb') as e:
        r=subprocess.run(cmd,input=prompt.encode('ascii'),cwd=d,stdout=f,stderr=e)
    rc=r.returncode
except Exception as e:exception=repr(e)
models=set();sessions=set();texts=[];results=[];parse_errors=[]
if(d/'events.jsonl').is_file():
    for lineno,line in enumerate((d/'events.jsonl').read_text(encoding='utf-8').splitlines(),1):
        try:x=json.loads(line)
        except Exception as e:parse_errors.append({'line':lineno,'error':repr(e)});continue
        if x.get('session_id'):sessions.add(x['session_id'])
        if x.get('type')=='assistant':
            m=x.get('message',{}).get('model')
            if m:models.add(m)
            texts.extend(c.get('text','') for c in x.get('message',{}).get('content',[]) if c.get('type')=='text')
        if x.get('type')=='result':results.append(x)
with(d/'assistant-text.txt').open('x',encoding='ascii') as f:
    f.write('\n'.join(texts).encode('ascii','backslashreplace').decode('ascii'))
packet_unchanged=packet.is_file() and sha(packet.read_bytes())==expected
v={'requested':model,'responding_models':sorted(models),'session_ids':sorted(sessions),
   'fresh_session_requested':True,'exit':rc,'elapsed_s':time.monotonic()-t,'launch_exception':exception,
   'source_packet_sha256':expected,'source_packet_unchanged':packet_unchanged,
   'result_events':results,'parse_errors':parse_errors,'retries':0}
with(d/'completion.json').open('x',encoding='ascii') as f:json.dump(v,f,indent=2,ensure_ascii=True);f.write('\n')
print(json.dumps({k:v[k] for k in ['requested','responding_models','session_ids','fresh_session_requested','exit','elapsed_s','launch_exception','source_packet_unchanged']},ensure_ascii=True),flush=True)
if rc is None or not packet_unchanged or parse_errors or not models:raise SystemExit(2)
raise SystemExit(rc)
