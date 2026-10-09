#!/usr/bin/env python3
"""Fail-closed raw batch observer coverage, not a numerical equivalence claim."""
import hashlib,math,struct
from pathlib import Path

def coverage(trace,expected_rids,stages,capture_root):
 expected=set(expected_rids);assert 1<=len(expected)<=6 and all(type(x) is int and x>0 for x in expected)
 assert stages and stages[0][1]==0 and stages[-1][2]==48
 for i,(stage,lb,le) in enumerate(stages):
  assert 0<=lb<le<=48 and (not i or stages[i-1][2]==lb)
 assert len({s for s,_,_ in stages})==len(stages)
 ownership={l:(stage,lb,le) for stage,lb,le in stages for l in range(lb,le)};root=Path(capture_root).resolve()
 requests={};resumes={};vectors={};spans=[];seen_paths=set();total=0;replays={};batch_events={};vector_replays={};replay_rows={};row_proofs={}
 for line in trace.splitlines():
  if not line.startswith('SBF '):continue
  kind=line.split()[1];fields=dict(word.split('=',1) for word in line.split()[2:] if '=' in word)
  if kind=='bound':raise AssertionError('Observer bound exceeded')
  if kind=='batch_event':
   key=(fields['pid'],int(fields['enginegen']),int(fields['event']));assert key not in batch_events and key[1]>0 and key[2]>0
   rows=int(fields['rows']);mask=int(fields['active_mask']);assert 1<=rows<=6 and 0<mask<64 and mask.bit_count()==rows and fields['completed']=='1'
   batch_events[key]=fields;continue
  if kind=='replay':
   stage,lb,le=map(int,(fields['stage'],fields['lb'],fields['le']));assert (stage,lb,le) in stages
   key=(fields['pid'],stage,int(fields['epoch']));assert key not in replays and key[2]>0 and 0<=int(fields['graph'])<16
   rows=int(fields['rows']);admission=int(fields['admission']);mask=int(fields['active_mask']);selected=int(fields['selected_mask']);event=int(fields['event'])
   assert fields['full_roster']==fields['input_output_verified']==fields['stage_context_verified']=='1' and fields['geometry']=='48x4x2560x248320'
   assert 1<=rows<=6 and mask>0 and mask<64 and mask.bit_count()==rows and 0<selected<(1<<rows)
   assert admission in (0,1) and (rows==1 and event==0 if admission else rows>=2 and event>0)
   replays[key]=fields;replay_rows[key]=set();continue
  if kind=='row':
   stage=int(fields['stage']);key=(fields['pid'],stage,int(fields['epoch']));assert key in replays and (stage,int(fields['lb']),int(fields['le'])) in stages
   proof=replays[key];row=int(fields['row']);rows=int(fields['batchrows']);selected=int(fields['selected'])
   assert 0<=row<rows==int(proof['rows']) and selected in (0,1) and int(proof['selected_mask'])&(1<<row)==selected*(1<<row)
   assert fields['enginegen']==proof['enginegen'] and fields['graph']==proof['graph'] and fields['event']==proof['event'] and fields['admission']==proof['admission']
   assert 0<=int(fields['slot'])<6 and int(fields['slotgen'])>0 and int(fields['pos'])>=0 and 0<=int(fields['token'])<248320
   if selected:assert int(fields['rid']) in expected and int(fields['requestgen'])>0
   rowkey=key+(row,);assert rowkey not in row_proofs;row_proofs[rowkey]=fields;continue
  if kind not in ('request','resume','vector','span'):continue
  rid=int(fields['rid']);assert rid in expected,'Unexpected RID';gen=int(fields['requestgen']);slotgen=int(fields['slotgen']);assert gen>0 and slotgen>0
  if kind=='request':
   assert rid not in requests,'Duplicate/replayed admission outside initial BGEN qualification scope'
   assert int(fields['enginegen'])>0
   requests[rid]=fields;continue
  assert rid in requests and fields['pid']==requests[rid]['pid'] and gen==int(requests[rid]['requestgen']) and int(fields['enginegen'])==int(requests[rid]['enginegen'])
  assert slotgen==int(requests[rid]['slotgen']) and fields['slot']==requests[rid]['slot'],'Unqualified migration/stale row identity'
  if kind=='resume':assert rid not in resumes;resumes[rid]=fields;continue
  stage,lb,le=map(int,(fields['stage'],fields['lb'],fields['le']));assert (stage,lb,le) in stages,'Wrong stage ownership'
  if kind=='span':
   a,b=int(fields['begin']),int(fields['end']);assert 0<=a<b and fields['completed']=='1';spans.append((rid,stage,fields['phase'],a,b));continue
  layer=int(fields['layer']);phase=fields['phase'];assert phase in ('admission_residual','admission_logits_before_sampler','batch_step_residual','batch_step_logits_before_sampler')
  admission=phase.startswith('admission_');pos=int(fields['pos']);prompt=int(requests[rid]['prompt']);rows=int(fields['batchrows']);row=int(fields['row'])
  assert 0<=row<rows<=6 and (rows==1 and pos==prompt-1 if admission else rows>=2 and pos>=prompt),'Missing genuine admission/later multi-row batch geometry'
  assert (stage,lb,le)==(ownership[layer] if layer>=0 else stages[-1])
  assert (layer>=0)==phase.endswith('residual') and (-1<=layer<48)
  count=248320 if layer==-1 else 10240;assert int(fields['floats'])==count and int(fields['bytes'])==count*4 and fields['canonical']=='le_f32'
  key=(rid,admission,layer);assert key not in vectors,'Duplicate vector'
  replay=(fields['pid'],stage,int(fields['epoch']));assert replay in replays,'Missing replay proof';proof=replays[replay]
  assert int(fields['graph'])==int(proof['graph']) and int(fields['enginegen'])==int(proof['enginegen']) and int(fields['event'])==int(proof['event'])
  assert int(proof['admission'])==int(admission) and int(proof['rows'])==rows and (int(proof['selected_mask'])&(1<<row)) and (int(proof['active_mask'])&(1<<int(fields['slot'])))
  rowproof=row_proofs[replay+(row,)]
  assert rowproof['selected']=='1' and all(fields[k]==rowproof[k] for k in ('rid','enginegen','requestgen','slot','slotgen','pos','token','row','batchrows','event','graph')),'Raw vector does not match exact GPU-verified row identity'
  replay_rows[replay].add((row,layer));vector_replays[key]=replay
  p=root/Path(fields['file']).name;assert p.resolve().parent==root and not p.is_symlink() and p not in seen_paths;seen_paths.add(p)
  raw=p.read_bytes();assert len(raw)==count*4 and all(math.isfinite(x[0]) for x in struct.iter_unpack('<f',raw));total+=len(raw);assert total<=64*1024*1024
  vectors[key]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'position':pos,'token':int(fields['token']),'stage':stage,'row':row,'batchrows':rows}
 assert set(requests)==set(resumes)==expected,'Missing request/resume records'
 assert len({(fields['pid'],fields['enginegen']) for fields in requests.values()})==1,'Mixed native engine generations'
 for replay,proof in replays.items():
  stage=int(proof['stage']);lb=int(proof['lb']);le=int(proof['le']);selected=int(proof['selected_mask']);rows=int(proof['rows']);mask=int(proof['active_mask'])
  expected_roster={(row,l) for row in range(rows) if selected&(1<<row) for l in list(range(lb,le))+([-1] if stage==stages[-1][0] else [])}
  assert replay_rows[replay]==expected_roster,'Empty/partial replay row/layer/head coverage'
  actual_rows=[row_proofs[replay+(row,)] for row in range(rows)];assert len({int(row['slot']) for row in actual_rows})==rows
  assert sum(1<<int(row['slot']) for row in actual_rows)==mask,'Wrong active slot row roster' 
  if not int(proof['admission']):
   event=(proof['pid'],int(proof['enginegen']),int(proof['event']));assert event in batch_events,'No actual completed later batch event'
   assert int(batch_events[event]['rows'])==rows and int(batch_events[event]['active_mask'])==mask,'Wrong actual batch event geometry'
 for rid in expected:
  for admission in (True,False):
   fields=[vectors[(rid,admission,l)] for l in range(-1,48) if (rid,admission,l) in vectors]
   assert len({(f['position'],f['token'],f['row'],f['batchrows']) for f in fields})==1,'Cross-stage row/token/position changed' 
 for rid in expected:
  for admission in (True,False):assert all((rid,admission,l) in vectors for l in range(-1,48)),'Missing full48 residuals or full head'
  start=int(resumes[rid]['read_from']);prompt=int(requests[rid]['prompt']);assert 0<=start<prompt
  # Count actual source spans separately for every stage. Prefix reuse is an observation of read_from, not a fullstate proof.
  for stage,_,_ in stages:
   actual=sorted((a,b) for r,s,phase,a,b in spans if r==rid and s==stage and phase in ('prefill_chunk','admission_target_verify') and a<prompt)
   at=start
   for a,b in actual:assert a==at and b<=prompt,'Missing/overlapping prompt evaluation spans';at=b
   assert at==prompt,'Incomplete per-stage prompt evaluation'
   later=[(a,b) for r,s,phase,a,b in spans if r==rid and s==stage and phase=='batch_target_verify']
   sample=vectors[(rid,False,-1)]['position'];assert any(a<=sample<b for a,b in later),'Later capture has no completed owning stage evaluation span'
 return {'status':'complete_raw_coverage_only','requests':sorted(expected),'vectors':len(vectors),'bytes':total,'vectors_by_identity':{str(k):v for k,v in vectors.items()},'spans':spans,'replay_proofs':len(replays),'completed_batch_events':len(batch_events),'scope':'initial BGEN admission plus actual later multi-row batch with exact per-stage replay epoch/roster proofs; no migration/cache/concurrency numerical or fairness qualification'}
