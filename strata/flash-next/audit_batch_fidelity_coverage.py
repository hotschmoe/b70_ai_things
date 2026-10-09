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
 requests={};resumes={};vectors={};spans=[];seen_paths=set();total=0
 for line in trace.splitlines():
  if not line.startswith('SBF '):continue
  kind=line.split()[1];fields=dict(word.split('=',1) for word in line.split()[2:] if '=' in word)
  if kind=='bound':raise AssertionError('Observer bound exceeded')
  if kind not in ('request','resume','vector','span'):continue
  rid=int(fields['rid']);assert rid in expected,'Unexpected RID';gen=int(fields['requestgen']);slotgen=int(fields['slotgen']);assert gen>0 and slotgen>0
  if kind=='request':
   assert rid not in requests,'Duplicate/replayed admission outside initial BGEN qualification scope'
   requests[rid]=fields;continue
  assert rid in requests and fields['pid']==requests[rid]['pid'] and gen==int(requests[rid]['requestgen'])
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
  p=root/Path(fields['file']).name;assert p.resolve().parent==root and not p.is_symlink() and p not in seen_paths;seen_paths.add(p)
  raw=p.read_bytes();assert len(raw)==count*4 and all(math.isfinite(x[0]) for x in struct.iter_unpack('<f',raw));total+=len(raw);assert total<=64*1024*1024
  vectors[key]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'position':pos,'token':int(fields['token']),'stage':stage,'row':row,'batchrows':rows}
 assert set(requests)==set(resumes)==expected,'Missing request/resume records'
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
 return {'status':'complete_raw_coverage_only','requests':sorted(expected),'vectors':len(vectors),'bytes':total,'vectors_by_identity':{str(k):v for k,v in vectors.items()},'spans':spans,'scope':'initial BGEN admission plus actual later multi-row batch; no migration/cache/concurrency numerical or fairness qualification'}
