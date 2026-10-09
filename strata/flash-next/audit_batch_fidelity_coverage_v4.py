#!/usr/bin/env python3
"""Bounded initial batch plus actual solo-migration raw producer coverage only."""
import hashlib,math,struct
from pathlib import Path
from audit_batch_fidelity_coverage_v2 import coverage as initial_coverage

def parsed(trace):
 for number,line in enumerate(trace.splitlines(),1):
  if line.startswith('SBF '):yield number,line.split()[1],dict(w.split('=',1) for w in line.split()[2:] if '=' in w),line

def coverage(trace,expected_rids,migration_rids,stages,capture_root):
 expected=set(expected_rids);migrating=set(migration_rids);assert migrating and migrating<=expected
 normal=[];records=list(parsed(trace));requests={};begins={};events={};closes={};replays={};rows={};vectors={};spans=[];resumes={};paths=set();total=0
 for number,kind,f,line in records:
  solo=kind in ('migration_begin','solo_event') or kind in ('replay','row') and f.get('admission')=='2' or kind=='vector' and f.get('phase','').startswith('solo_migration_') or kind in ('span','resume') and f.get('slot')=='6'
  if not solo:normal.append(line)
  if kind=='request':requests[int(f['rid'])]=f
  if kind=='slot_closed':
   rid=int(f['rid']);assert rid in expected and f['completed']=='1'
   key=(rid,int(f['slotgen']));assert key not in closes,'Duplicate actual slot close';closes[key]=(number,f)
  if kind=='migration_begin':
   rid=int(f['rid']);assert rid in migrating and rid not in begins and rid in requests
   old=requests[rid];assert all(f[k]==old[k] for k in ('pid','enginegen','requestgen'))
   assert f['slot']=='6' and f['main_session']=='1' and f['source_slot']==old['slot'] and f['source_slotgen']==f['slotgen']==old['slotgen']
   assert int(f['prompt'])>int(old['prompt']) and int(f['event'])==int(f['requestgen'])>0
   close_at,closed=closes[(rid,int(f['slotgen']))];assert close_at<number and closed['cancelled']=='1' and closed['slot']==old['slot'] and all(closed[k]==f[k] for k in ('pid','enginegen','requestgen'))
   begins[rid]=(number,f)
  if kind=='solo_event':
   rid=int(f['rid']);assert rid in begins and rid not in events and f['completed']==f['main_session']==f['rows']=='1'
   assert all(f[k]==begins[rid][1][k] for k in ('pid','enginegen','requestgen','event')) and int(f['pos'])==int(begins[rid][1]['prompt'])-1
   events[rid]=(number,f)
  if kind in ('replay','row') and f.get('admission')=='2':
   stage=int(f['stage']);assert (stage,int(f['lb']),int(f['le'])) in stages
   key=(f['pid'],stage,int(f['epoch']));assert key[2]>0 and 0<=int(f['graph'])<16
   if kind=='replay':
    assert key not in replays and f['rows']=='1' and f['active_mask']=='64' and f['selected_mask']=='1'
    assert f['full_roster']==f['input_output_verified']==f['stage_context_verified']=='1' and f['geometry']=='48x4x2560x248320';replays[key]=(number,f)
   else:
    assert key in replays and key not in rows and f['row']=='0' and f['batchrows']=='1' and f['selected']=='1' and f['slot']=='6'
    rid=int(f['rid']);assert rid in begins and begins[rid][0]<number
    assert all(f[k]==begins[rid][1][k] for k in ('pid','enginegen','requestgen','slotgen','event')) and int(f['pos'])==int(begins[rid][1]['prompt'])-1
    assert f['graph']==replays[key][1]['graph'] and f['event']==replays[key][1]['event'];rows[key]=(number,f)
  if kind=='resume' and f.get('slot')=='6':
   rid=int(f['rid']);assert rid in begins and rid not in resumes and all(f[k]==begins[rid][1][k] for k in ('pid','enginegen','requestgen','slotgen'));resumes[rid]=f
  if kind=='span' and f.get('slot')=='6':
   rid=int(f['rid']);assert rid in begins and f['completed']=='1' and all(f[k]==begins[rid][1][k] for k in ('pid','enginegen','requestgen','slotgen'))
   assert (int(f['stage']),int(f['lb']),int(f['le'])) in stages and 0<=int(f['begin'])<int(f['end'])
   spans.append(f)
  if kind=='vector':
   file=Path(capture_root).resolve()/Path(f['file']).name;assert file not in paths and file.resolve().parent==Path(capture_root).resolve() and not file.is_symlink();paths.add(file);total+=int(f['bytes']);assert total<=64<<20
   if f.get('phase','').startswith('solo_migration_'):
    rid=int(f['rid']);layer=int(f['layer']);assert rid in begins and -1<=layer<48
    key=(f['pid'],int(f['stage']),int(f['epoch']));assert key in rows;proof=rows[key][1]
    assert all(f[k]==proof[k] for k in ('rid','enginegen','requestgen','slot','slotgen','pos','token','row','batchrows','event','graph'))
    assert (f['phase']=='solo_migration_residual')==(layer>=0)
    owner=next((s,lb,le) for s,lb,le in stages if lb<=layer<le) if layer>=0 else stages[-1];assert owner==(int(f['stage']),int(f['lb']),int(f['le']))
    count=248320 if layer==-1 else 10240;assert f['canonical']=='le_f32' and int(f['floats'])==count and int(f['bytes'])==count*4
    raw=file.read_bytes();assert len(raw)==count*4 and all(math.isfinite(v[0]) for v in struct.iter_unpack('<f',raw))
    identity=(rid,layer);assert identity not in vectors;vectors[identity]={'path':str(file),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'position':int(f['pos']),'token':int(f['token']),'proof_key':key,'line':number}
 initial=initial_coverage('\n'.join(normal),expected_rids,stages,capture_root)
 assert set(begins)==set(events)==set(resumes)==migrating
 for rid in migrating:
  assert all((rid,l) in vectors for l in range(-1,48))
  assert len({(vectors[rid,l]['position'],vectors[rid,l]['token']) for l in range(-1,48)})==1
  prompt=int(begins[rid][1]['prompt']);start=int(resumes[rid]['read_from']);assert 0<=start<prompt and 0<=int(resumes[rid]['reused'])<=prompt and int(resumes[rid]['reread_to'])==-1
  for stage,lb,le in stages:
   actual=sorted((int(s['begin']),int(s['end'])) for s in spans if int(s['rid'])==rid and int(s['stage'])==stage and s['phase'] in ('prefill_chunk','solo_migration_target_verify') and int(s['begin'])<prompt);at=start
   for a,b in actual:assert a==at and b<=prompt;at=b
   assert at==prompt,'Missing/overlapping actual solo per-stage evaluated prefix spans'
   chosen=[(k,n,f) for k,(n,f) in rows.items() if int(f['rid'])==rid and k[1]==stage];assert len(chosen)==1
   key,at,f=chosen[0];required=set(range(lb,le))|({-1} if stage==stages[-1][0] else set());actual={l for (r,l),v in vectors.items() if r==rid and v['proof_key']==key};assert actual==required
   assert all(vectors[rid,l]['line']<events[rid][0] for l in required)
   pos=vectors[rid,-1]['position'];assert any(int(s['rid'])==rid and int(s['stage'])==stage and s['phase']=='solo_migration_target_verify' and int(s['begin'])<=pos<int(s['end']) for s in spans)
 return {'status':'complete_raw_coverage_only','initial':initial,'migration_requests':sorted(migrating),'migration_vectors':len(vectors),'total_raw_bytes':total,'migration_vectors_by_identity':{str(k):v for k,v in vectors.items()},'source_slot_close_and_real_solo_body_event':True,'unobserved_actual_migration_math':True,'full_model_math_qualified':False,'scope':'source callbacks + GPU-verified raw main-session all48/head after actual slot cancel/solo GEN; independent prefix/serial comparison and original state transfer/math/allocator/source gates remain separate'}
