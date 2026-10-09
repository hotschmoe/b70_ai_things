#!/usr/bin/env python3
"""Exact tracked22 reconstruction, actual API methods over CPU socketpair + native guard source extraction."""
import ast,hashlib,importlib.util,json,os,queue,socket,subprocess,tempfile,threading,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];plan=json.loads((ROOT/'strata/flash-next/batch-request-identity-draft-plan-v2.json').read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def reject(fn):
 try:fn()
 except (ValueError,RuntimeError):return
 raise AssertionError('Stale identity accepted')
with tempfile.TemporaryDirectory(prefix='batch-transport-cpu-') as name:
 w=Path(name);base=Path(plan['base'])
 for rel,h in plan['base_files'].items():
  assert sha(base/rel)==h;p=w/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((base/rel).read_bytes())
 patch=ROOT/plan['patch'];assert sha(patch)==plan['patch_sha256'];subprocess.run(['git','apply','--check',str(patch)],cwd=w,check=True);subprocess.run(['git','apply',str(patch)],cwd=w,check=True)
 for rel,h in plan['overlays'].items():assert sha(w/rel)==h
 spec=importlib.util.spec_from_file_location('identity',w/'serve/batch_request_identity.py');identity=importlib.util.module_from_spec(spec);spec.loader.exec_module(identity)
 tree=ast.parse((w/'serve/server.py').read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='StrataEngine')
 names={'last','_begin_request_identity','_request_keys','_stop_slot_identity','_parse_done','sampling_keys','projection_key','_send','_pump','_control','_drain_control','_release_slot_when_done','alive','exit_code','_may_go_solo','_shorter_waiting','pick_slot','generate_batched','_take_control'}
 body=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in names or isinstance(n,ast.Assign) and (isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='RequestField' or any(isinstance(t,ast.Name) and t.id in ['SOLO_AGAIN_MAX','SOLO_AGAIN_MIN_LEFT','SLOT_PREFIX_MIN','YIELDS_MAX'] for t in n.targets))]
 module=ast.Module(body=[ast.ClassDef(name='Engine',bases=[],keywords=[],body=body,decorator_list=[])],type_ignores=[]);ast.fix_missing_locations(module)
 ns={'os':os,'threading':threading,'time':time,'queue':queue,'RequestField':identity.RequestField,'batch_identity_tags':identity.tags,'EngineDied':RuntimeError,'EOS_IDS':set(),'FATAL_PREFIXES':('FATAL ',),'btrace':lambda *a:None};exec(compile(module,'verified22-server-AST','exec'),ns);Engine=ns['Engine'];ns['StrataEngine']=Engine
 e=Engine();e.batch=2;e.strict_batch_identity=True;e.gen=1;e._request_identity_lock=threading.Lock();e._request_identity_counter=0;e.slot_identity=[None,None];e._begin_request_identity();rid=e._tl.rid;e.wlock=threading.Lock();e.lines=queue.Queue();e.slot_q=[queue.Queue(),queue.Queue()];e.slot_cv=threading.Condition();e.slot_busy=[True,False];e.slot_held=[[],[]];e.slot_used=[0,0];e.slot_order=[0,1];e.slot_group=[0,0];e.slot_groups=1;e.waiting=0;e.info={'slot_cache':'1'};e.ended=False
 client,native=socket.socketpair();cin=client.makefile('w');cout=client.makefile('r');nin=native.makefile('r');nout=native.makefile('w');e.proc=type('Proc',(),{'stdin':cin,'stdout':cout,'poll':lambda self:None})()
 pump=threading.Thread(target=e._pump,daemon=True);pump.start();seen=[]
 def producer():
  for line in nin:
   seen.append(line.strip())
   if line.startswith('GEN '):
    nout.write('RESUME 5\nPP 10 20 1 100\nYIELDED 0 10 rid=%d slotgen=1\nDONE 0 20 1 0 cancel 0 0 5 0 0 0 0 0 5 0 rid=%d slotgen=0\n'%(rid,rid));nout.flush()
   elif line.startswith('BGEN '):
    nout.write('T 17 rid=%d slotgen=2\nDONE 1 20 1 2 length 0 0 10 0 0 0 0 0 10 0 rid=%d slotgen=2\nBADM 0 1 rid=%d slotgen=2\n'%(rid,rid,rid));nout.flush()
   elif line.startswith('QUIT'):break
 worker=threading.Thread(target=producer,daemon=True);worker.start();cancel=threading.Event();tokens=[]
 e._ctl_mode='solo';e._send('GEN 64'+e._request_keys({})+' 11,12,13');list(e._control(cancel,tokens.append));assert e._yielded==(0,10) and e.reused==5 and e.last['request_id']==rid
 # BYIELD continuation keeps request identity, advances native slot generation.
 e._ctl_mode='batch';e._send('BGEN 0 64'+e._request_keys({})+' 11,12,13');list(e._control(cancel,tokens.append));assert tokens==[17] and e.slot_identity[0]==(1,rid,2) and len(e.last['segments'])==2
 # Solo migration uses the same RID and active slot-aware cancellation.
 assert e._may_go_solo(40,0,None);e._stop_slot_identity(0);time.sleep(.02);assert any('BSTOP 0 rid=%d slotgen=2'%rid in x for x in seen)
 # Detached drain must not retire on a prior generation's BDONE.
 e._release_slot_when_done(0,[11,12,13,17]);nout.write('BT 0 99 rid=%d slotgen=1\nBDONE 0 2 cancel 1 rid=%d slotgen=1\n'%(rid,rid));nout.flush();time.sleep(.05);assert e.slot_busy[0]
 nout.write('BT 0 18 rid=%d slotgen=2\nBDONE 0 2 cancel 1 rid=%d slotgen=2\n'%(rid,rid));nout.flush()
 deadline=time.monotonic()+2
 while e.slot_busy[0] and time.monotonic()<deadline:time.sleep(.01)
 assert not e.slot_busy[0] and 99 not in e.slot_held[0]
 # Restart invalidates stop/drain identity; neither can touch the new generation.
 e.gen=2;reject(lambda:e._stop_slot_identity(0));assert e._drain_control('DONE',timeout=.01,born=1) is None
 e.gen=1;e._ctl_mode='solo';old_queue=e.lines
 nout.write('PP 1 20 1 1\n');nout.flush();reader=e._control(cancel,tokens.append);next(reader)
 fresh_queue=queue.Queue();fresh_queue.put('T 99 rid=999 slotgen=0');e.lines=fresh_queue;e.gen=2
 reject(lambda:next(reader));assert fresh_queue.qsize()==1
 e.gen=1;e.lines=old_queue
 for message in ['T 99 rid=999 slotgen=0','YIELDED 1 10 rid=999 slotgen=7']:
  nout.write(message+'\n');nout.flush();before=list(tokens);reject(lambda:list(e._control(cancel,tokens.append)));assert tokens==before
 e.gen=2
 e._send('QUIT');worker.join(timeout=1);client.shutdown(socket.SHUT_RDWR);client.close();native.close()
 # Exercise actual generator migration and BYIELD loops, not hand-copied transitions.
 for scenario in ['migration','yield']:
  e=Engine();e.batch=2;e.strict_batch_identity=True;e.gen=1;e._request_identity_lock=threading.Lock();e._request_identity_counter=0;e.slot_identity=[None,None];e._begin_request_identity();rid=e._tl.rid;e.wlock=threading.Lock();e.lines=queue.Queue();e.slot_q=[queue.Queue(),queue.Queue()];e.slot_cv=threading.Condition();e.slot_busy=[False,False] if scenario=='yield' else [False,True];e.slot_held=[[],[]];e.slot_live=[None,None];e.slot_used=[0,0];e.slot_order=[0,1];e.slot_group=[0,0];e.slot_groups=1;e.waiting=0;e.wait_lens=[];e.ctl_epoch=0;e.ctl=threading.Lock();e.info={'slot_cache':'1'};e.ended=False;e.progress=None;e.progress_ms=0;e.reused=0;e._ctl_erred=False;e._yielded=None;e._ctl_mode=None;e._ctl_result=None
  client,native=socket.socketpair();cin=client.makefile('w');cout=client.makefile('r');nin=native.makefile('r');nout=native.makefile('w');e.proc=type('Proc',(),{'stdin':cin,'stdout':cout,'poll':lambda self:None})();seen=[]
  pump=threading.Thread(target=e._pump,daemon=True);pump.start()
  def migration_producer():
   for line in nin:
    seen.append(line.strip())
    if line.startswith('BGEN '):
     nout.write('T 17 rid=%d slotgen=1\nDONE 1 20 1 2 length 0 0 5 0 0 0 0 0 15 0 rid=%d slotgen=1\nBADM 0 1 rid=%d slotgen=1\n'%(rid,rid,rid));nout.flush()
     with e.slot_cv:e.slot_busy[1]=False
     nout.write('BT 0 18 rid=%d slotgen=1\n'%rid);nout.flush()
     if scenario=='yield':nout.write('BDONE 0 2 stop 1 rid=%d slotgen=1\n'%rid);nout.flush()
    elif line.startswith('BSTOP '):nout.write('BDONE 0 2 cancel 1 rid=%d slotgen=1\n'%rid);nout.flush()
    elif line.startswith('BYIELD '):
     with e.slot_cv:e.waiting=0;e.wait_lens=[]
     nout.write('YIELDED 0 10 rid=%d slotgen=1\nDONE 0 20 1 0 cancel 0 0 0 0 0 0 0 0 10 0 rid=%d slotgen=0\n'%(rid,rid));nout.flush()
    elif line.startswith('GEN ') and scenario=='yield':
     with e.slot_cv:e.waiting=1;e.wait_lens=[[1]]
     nout.write('PP 10 20 1 10\n');nout.flush()
    elif line.startswith('GEN '):nout.write('T 19 rid=%d slotgen=0\nDONE 1 22 1 2 stop 0 0 20 0 0 0 0 0 2 0 rid=%d slotgen=0\n'%(rid,rid));nout.flush()
    elif line.startswith('QUIT'):break
  worker=threading.Thread(target=migration_producer,daemon=True);worker.start()
  actual=[x for x in e.generate_batched([11,12,13],2 if scenario=='yield' else 40,{},threading.Event()) if x is not None]
  assert actual==([17,18] if scenario=='yield' else [17,18,19]) and any(x.startswith('BGEN ') for x in seen) and any(x.startswith('BYIELD ' if scenario=='yield' else 'BSTOP ') for x in seen) and any(x.startswith('GEN ') for x in seen)
  assert all(' rid='+str(rid) in x for x in seen if x.startswith(('BGEN ','GEN ','BSTOP '))) and not e.slot_busy[0]
  e._send('QUIT');worker.join(timeout=1);client.shutdown(socket.SHUT_RDWR);client.close();native.close()
 # Compile exact native BSTOP closure extracted from hashed patched source.
 cpp=(w/'sycl/src/program/generate.cpp').read_text();a=cpp.index('        auto slot_stop =');b=cpp.index('        auto batch_on =',a);stop=cpp[a:b]
 rid_start=cpp.index('                    if(key=="rid")');rid_end=cpp.index('                    else if (key == "cvec")',rid_start);rid_code=cpp[rid_start:rid_end]
 cap_start=cpp.index('    const int requested_batch=o.batch;');cap_end=cpp.index('    std::vector<std::vector<std::unique_ptr<',cap_start);cap_code=cpp[cap_start:cap_end]
 fit_start=cpp.index('        if(strict_batch_identity && fit!=requested_batch)');fit_end=cpp.index('        for (auto& v : bslot_ss)',fit_start);fit_code=cpp[fit_start:fit_end]
 native_guard='''#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <string>
#include <vector>
#include <cassert>
namespace strata::core {bool native_hc_requested(){return true;}}
struct Options{int batch=2,batch_groups=1,pipeline_windows=0;std::string mtp;};
int capacity(Options o,int fit){bool batch_mtp=false;
'''+cap_code+fit_code+'''return 0;}
bool request_id(const std::string& tok){auto eq=tok.find('=');auto key=tok.substr(0,eq);uint64_t req_rid=0;bool req_rid_bad=false;
'''+rid_code+'''return req_rid>0&&!req_rid_bad;}
struct Slot{uint64_t request_id=7,generation=3;bool stop=false;};
int main(){
for(int slots:{2,4,6}){Options o;o.batch=slots;assert(capacity(o,slots)==0);assert(capacity(o,slots-1)==2);}
for(int slots:{1,3,8}){Options o;o.batch=slots;assert(capacity(o,slots)==2);}
assert(request_id("rid=7"));for(const auto* value:{"rid=0","rid=-1","rid=x","rid=7x","rid=18446744073709551616"})assert(!request_id(value));
std::vector<Slot> bs(2);bool strict_batch_identity=true;
'''+stop+'''
slot_stop("BSTOP 0 rid=7 slotgen=3");assert(bs[0].stop&&!bs[1].stop);bs[0].stop=false;
for(const auto* text:{"BSTOP 0","BSTOP 0 rid=8 slotgen=3","BSTOP 0 rid=7 slotgen=2","BSTOP 0 rid=7x slotgen=3","BSTOP 0 rid=7 slotgen=3x","BSTOP 0 rid=18446744073709551616 slotgen=3"}){slot_stop(text);assert(!bs[0].stop&&!bs[1].stop);}
std::puts("PASS exact native capacity/RID/BSTOP guards; stale/malformed/overflow identity cannot stoppeer");}
'''
 (w/'guard.cpp').write_text(native_guard,encoding='ascii')
 cmd=['docker','run','--rm','--network','none','--entrypoint','/bin/bash','-v',str(w)+':/work','sha256:39992d7072aa0557f4e3a5faf7782f83bab8fe4882e9d8fcf3e660d3de50e9e7','-lc','g++ -std=c++20 -O1 -fsanitize=address,undefined -fno-sanitize-recover=all /work/guard.cpp -o /work/guard && /work/guard']
 result=subprocess.run(cmd,capture_output=True,text=True,check=True);print(result.stdout.strip())
 print('PASS actual patched API socket/pump/control, BYIELD/admission identity, sameRID migration, detached stale-drain and restart invalidation; tracked-source binding. CPU only/no model/math/GPU qualification.')
