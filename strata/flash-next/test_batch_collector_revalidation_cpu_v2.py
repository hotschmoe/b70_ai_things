#!/usr/bin/env python3
"""Actual immutable batch/serial collectors on bounded synthetic raw fields."""
import hashlib,json,re,shutil,struct,tempfile
from pathlib import Path
import audit_batch_numerical_suite_v2 as audit
import serial_prefix_qualification_v6 as collector
from audit_batch_fidelity_coverage_v3 import coverage
HERE=Path(__file__).resolve().parent

def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
def reject(fn):
 try:fn()
 except (AssertionError,ValueError,KeyError):return
 raise AssertionError('Mutated original collector provenance accepted')

def main():
 controls=0
 with tempfile.TemporaryDirectory() as name:
  root=Path(name);batch=root/'batch';child=batch/'child';cap=child/'captures';cap.mkdir(parents=True)
  latest=json.loads(Path('/mnt/vm_8tb/b70/build/strata-observer26-cpu-latest-v1.json').read_bytes());fixture=Path(latest['output']);trace=(fixture/'trace.log').read_text();assert hashlib.sha256(trace.encode()).hexdigest()==latest['trace_sha256']
  for line in trace.splitlines():
   if not line.startswith('SBF vector '):continue
   f=dict(re.findall(r'(\w+)=([^ ]+)',line));source=fixture/'capture'/Path(f['file']).name;shutil.copyfile(source,cap/source.name);trace=trace.replace(f['file'],'/results/captures/'+source.name)
  write(batch/'input-plan.snapshot.json',{'kind':'api','stage_ranges':[[0,32],[32,48]]});(child/'engine.combined.log').write_text(trace);saved=coverage(trace,[11,12],[11,12],[(0,0,32),(1,32,48)],cap);write(child/'report.json',{'raw':{'raw':saved}})
  actual=audit.vectors(batch);assert len(actual)==294
  first=Path(next(iter(actual.values())));original=first.read_bytes();first.write_bytes(struct.pack('<f',123.)+original[4:]);reject(lambda:audit.vectors(batch));controls+=1;first.write_bytes(original)
  fake=trace.replace('/results/captures/','/foreign/',1);(child/'engine.combined.log').write_text(fake);reject(lambda:audit.vectors(batch));controls+=1;(child/'engine.combined.log').write_text(trace)
  outside=root/'outside.bin';outside.write_bytes(original);first.unlink();first.symlink_to(outside);reject(lambda:audit.vectors(batch));controls+=1;first.unlink();first.write_bytes(original)
  serial=root/'serial';scap=serial/'child/serial-0/captures';scap.mkdir(parents=True);ids=[1,2];job={'rid':11,'role':'admission','ids':ids,'position':1,'token':2};write(child/'serial-jobs.json',{'jobs':[job]});write(serial/'input-plan.snapshot.json',{'group_index':0,'batch_parent':str(batch),'stage_ranges':[[0,48]]})
  led=[{'event':'selection','stages':1},{'event':'evaluated_span','lo':0,'hi':2,'prompt_rows':2},{'event':'finish','prompt_tokens':2,'evaluated_prompt_rows':2,'actual_reused':0,'stage_prompt_rows':2,'cancelled':False}]
  lines=['PREFIX_DIAG '+json.dumps(x) for x in led];lines+=['SFD request pid=1 request=1 tokens=2 ids=1,2']
  for layer in range(-1,48):
   count=248320 if layer==-1 else 10240;file=scap/(str(layer)+'.bin');file.write_bytes(struct.pack('<f',0.)*count);phase='first_logits_before_sampler' if layer==-1 else 'first_window_residual';lines.append(f'SFD vector pid=1 request=1 phase={phase} stage=0 lb=0 le=48 layer={layer} pos=1 token=2 bytes={count*4} nonfinite=0 file=/results/captures/{file.name}')
  common={'pid':1,'request':1};begin=dict(common,event='begin',input_sha256_le32=collector.le32_digest(ids),tokens=2);commit=dict(common,event='committed_live',ids_truncated=False,tokens=3,ids=ids+[0],sha256_le32=collector.le32_digest(ids+[0]),published=True,chain_updated=True,live_reusable=True,phase='complete')
  spans=[dict(common,event='stage_span',device=0,lb=0,le=48,lo=0,hi=2,phase='verify',complete=x) for x in (False,True)]
  lines+=['PCL '+json.dumps(x) for x in [begin,*spans,commit]];raw={'ids':ids,'fresh':1,'output_ids':[0],'stderr':lines};meta=collector.extract(raw,scap,True,True,[(0,48)]);requests=serial/'child/serial-0/requests.json';write(requests,[{'job':job,'raw':raw,'meta':meta}]);actual_serial=audit.serial_vectors(serial);assert len(actual_serial)==49
  changed=scap/'0.bin';original_serial=changed.read_bytes();changed.write_bytes(struct.pack('<f',123.)+original_serial[4:]);reject(lambda:audit.serial_vectors(serial));controls+=1;changed.write_bytes(original_serial)
  rows=json.loads(requests.read_bytes());rows[0]['job']['token']=9;write(requests,rows);reject(lambda:audit.serial_vectors(serial));controls+=1
 print(json.dumps({'passed':True,'synthetic_batch_vectors_rechecked_by_original_collector':294,'synthetic_serial_vectors_rechecked_by_original_collector':49,'negative_controls':controls,'scope':'Original source collectors on synthetic CPU raw fixtures only; no GPU/weights/model math/genuine serving qualification'},ensure_ascii=True))
if __name__=='__main__':main()
