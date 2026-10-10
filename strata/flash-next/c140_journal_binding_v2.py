"""Original C137 parent journal/health evidence admission, metadata only."""
import math,re
from pathlib import Path
import c1_serve_controller_combined_v140_v2 as c
FAULT=re.compile(r'Fault response|CAT error|GPU HANG|GPU coredump|Job .* timed out|GT.* reset failed',re.I)
HEALTH='sha256:d55637b3353eaf470677627dc1627c3dda3ec6a6abb0298451aed34b73937067'
def journal_binding(root,proof):
 root=Path(root).resolve();rows=proof['kernel_journal_receipts'];c.require(set(rows)=={'pre','active','post'} and rows['active'],'Exact pre/active/post original journal roster required')
 c.require(all(type(proof[k]) in (int,float) and math.isfinite(proof[k]) and proof[k]>0 for k in ('started_epoch','launch_started_epoch','terminal_epoch','pre_health_finished_epoch','post_health_finished_epoch','identity_started_epoch','journal_binding_epoch')),'Actual global journal chronology nonfinite/invalid')
 for stage in ('pre','post'):
  row=rows[stage];c.require(row==c.read(root/(stage+'-kernel-receipt.json')),'Original journal sidecar changed/missing')
  c.require(row['started_epoch']>=proof[stage+'_health_finished_epoch'],'Journal before complete health')
  if stage=='pre':c.require(row['finished_epoch']<=proof['launch_started_epoch'],'Prejournal mustfinish before actual launch')
  else:c.require(row['started_epoch']>=proof['terminal_epoch'] and proof['identity_started_epoch']>=row['finished_epoch'],'Postjournal/terminal/new4 chronology differs')
 allrows=[rows['pre'],*rows['active'],rows['post']];names=[]
 c.require(1<=len(rows['active'])<=320,'Bounded actual readiness journal rows required')
 for i,row in enumerate(allrows):
  label='pre-kernel-journal' if i==0 else 'post-kernel-journal' if i==len(allrows)-1 else 'active-kernel-journal-'+str(i)
  names.append(label);path=root/(label+'.log');cmd=root/(label+'.command.json');argv=['journalctl','-k','--since','@'+str(int(proof['started_epoch'])),'--no-pager']
  c.require(row['path']==str(path) and row['command']==argv==c.read(cmd) and row['command_file_sha256']==c.sha(cmd) and row['sha256']==row['stdout_sha256']==c.sha(path) and row['return_code']==0 and row['error'] is None,'Original complete journal command/log/rc/error differs')
  c.require(all(type(row[k]) in (int,float) and math.isfinite(row[k]) for k in ('started_epoch','finished_epoch')) and proof['started_epoch']<=row['started_epoch']<=row['finished_epoch']<=proof['journal_binding_epoch'],'Original journal finite chronology invalid')
  if 0<i<len(allrows)-1:c.require(proof['launch_started_epoch']<=row['started_epoch']<=row['finished_epoch']<=proof['terminal_epoch'],'Activejournal outside actual launch/terminal interval')
  c.require(not FAULT.search(path.read_text()),'Actual journal fault signature')
 c.require(all(a['finished_epoch']<=b['started_epoch'] for a,b in zip(allrows,allrows[1:])),'Original sequential journal command ordering differs')
 return rows
def health_binding(root,proof,stage):
 root=Path(root).resolve();p=root/(stage+'-health.json');h=c.read(p)
 c.require(h['passed'] is True and h['cards']==[0,1] and h['health_image']==HEALTH and h['finished_epoch']==proof[stage+'_health_finished_epoch'] and c.sha(p)==proof[stage+'_health_sha256'] and len(h['files'])==2,'Original exact pairhealth receipt differs')
 c.require(h['started_epoch']>=(proof['started_epoch'] if stage=='pre' else proof['terminal_epoch']),'Actual prehealth before parentstart or posthealth before terminal')
 commands=[[str(c.REPO/'vllm/int4/diagnostics/xpu_health_strict.sh'),'--img',HEALTH],[str(c.REPO/'bin/xpu-collective-health'),'--img',HEALTH,'--p2p','0','--timeout','180']]
 for row,argv,label in zip(h['files'],commands,('health.log','collective.log')):
  path=root/(stage+'-'+label);cmd=root/(stage+'-'+label+'.command.json')
  c.require(row['path']==str(path) and row['command']==argv==c.read(cmd) and row['command_file_sha256']==c.sha(cmd) and row['sha256']==row['stdout_sha256']==c.sha(path) and row['return_code']==0 and row['error'] is None,'Original strict/compiled health command/rc/error/log differs')
  c.require(all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in (h['started_epoch'],h['finished_epoch'],row['started_epoch'],row['finished_epoch'])),'Original health timestamps invalid')
  c.require(h['started_epoch']<=row['started_epoch']<=row['finished_epoch']<=h['finished_epoch'],'Original health command chronology differs')
 return h
def parent_binding(root,proof,identity,parent=None):
 """Bind actual full4 timing and original producer rows, not duplicated labels."""
 root=Path(root).resolve();path=root/'parent-before-proof.json';original=c.read(path)
 c.require(c.sha(path)==proof['parent_before_proof_sha256'] and original['parent_generation']==1402 and original['parent_controller_sha256']==proof['parent_controller_sha256'],'Original producer parent snapshot identity differs')
 fields={'started_epoch':'started_epoch','launch_started_epoch':'launch_started_epoch','pre_health_finished_epoch':'pre_health_finished_epoch','post_health_finished_epoch':'post_health_finished_epoch','terminal_epoch':'owned_terminal_epoch','journal_binding_epoch':'proof_binding_epoch','kernel_journal_receipts':'kernel_journal_receipts','engine_receipt_sha256':'engine_receipt_sha256','prepared_sha256':'prepared_sha256'}
 c.require(all(proof[a]==original[b] for a,b in fields.items()),'Proof journal/launch/health/parent fields differ from original producer')
 c.require(proof['identity_started_epoch']==identity['started']==original['post_model_identity']['started'] and identity['finished']==original['post_model_identity']['finished'] and original['post_model_identity']['sha256']==proof['identity']['sha256'] and original['post_model_identity']['passed'] is True,'Actual full4 identity.started/finished/hash differs from parent/proof')
 if parent is not None:
  c.require(parent['parent_generation']==1402 and parent['parent_controller_sha256']==original['parent_controller_sha256'] and all(parent[b]==original[b] for b in fields.values()) and parent['post_model_identity']==original['post_model_identity'],'Final actual parent differs from original journal/launch/health/identity snapshot')
 return original
