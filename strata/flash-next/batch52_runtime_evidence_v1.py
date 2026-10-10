"""Read-only original journal/child ownership bindings; no device/model access."""
import math,re
from pathlib import Path
from batch_numerical_proofs_v52 import read,sha,require
FAULT=re.compile(r'Fault response|CAT error|GPU HANG|GPU coredump|Job .* timed out|GT.* reset failed',re.I)
def journal_binding(root,parent,stage):
 root=Path(root).resolve();require(stage in ('pre','post'),'Exact journal stage required')
 require(all(type(parent[k]) in (int,float) and math.isfinite(parent[k]) and parent[k]>0 for k in ('started_epoch','finished_epoch','child_started_epoch','child_terminal_epoch','pre_health_finished_epoch','post_health_finished_epoch')),'Original finite parent chronology required')
 row=parent['kernel_journal_receipts'][stage];require(row==read(root/(stage+'-kernel-receipt.json')),'Actual original journal receipt changed/missing')
 path=root/(stage+'-kernel-journal.log');cmd=root/(stage+'-kernel-journal.command.json');argv=['journalctl','-k','--since','@'+str(int(parent['started_epoch'])),'--no-pager']
 require(row['return_code']==0 and row['error'] is None and row['path']==str(path) and row['command']==argv==read(cmd) and row['command_file_sha256']==sha(cmd) and row['sha256']==row['stdout_sha256']==sha(path),'Original journal rc/error/argv/log binding differs')
 require(all(type(row[k]) in (int,float) and math.isfinite(row[k]) for k in ('started_epoch','finished_epoch')) and parent['started_epoch']<=row['started_epoch']<=row['finished_epoch']<=parent['finished_epoch'],'Journal chronology invalid')
 require(row['started_epoch']>=parent[stage+'_health_finished_epoch'],'Journal before complete stagehealth')
 if stage=='pre':require(parent['child_started_epoch']<=parent['handshake_ready_observed_epoch']<=row['started_epoch']<=row['finished_epoch']<=parent['handshake_ack_epoch']<=parent['child_terminal_epoch'],'Actual post-READY prejournal/ACK chronology differs')
 else:require(row['started_epoch']>=parent['child_terminal_epoch'],'Postjournal before child terminal')
 require(not FAULT.search(path.read_text()),'Actual kernel fault signature present');return row
def child_binding(root,parent,plan,child):
 root=Path(root).resolve();here=Path(__file__).resolve().parent
 from serial37_stdout_capture_v2 import binding as stdout_binding
 stdout_binding(root,parent)
 from batch52_health_handshake_v1 import saved_binding
 saved_binding(root,parent,plan,child)
 require(parent.get('parent_generation')==52,'Original V7/older parent evidence cannot transfer')
 require(all(type(parent[k]) in (int,float) and math.isfinite(parent[k]) and parent[k]>0 for k in ('child_started_epoch','child_terminal_epoch','leaf_logs_epoch','post_health_finished_epoch')),'Original child/log chronology invalid')
 require((root/'wrapper.py').read_bytes()==(here/'qualify_batch_numerical_v52.py').read_bytes() and (root/'controller.py').read_bytes()==(here/'batch_numerical_execution_v52.py').read_bytes(),'Original child/parent source snapshots changed')
 interpreter=parent['child_interpreter'];require(Path(interpreter['path']).resolve()==Path(interpreter['path']) and sha(interpreter['path'])==interpreter['sha256'],'Original child interpreter changed')
 command=read(root/'child.command.json');require(command==[interpreter['path'],str(here/'batch_numerical_execution_v52.py'),'run','--plan',str((root/'admitted-plan.snapshot.json').resolve()),'--pre-health',str(root/'pre-health.json'),'--handshake',str(root/'handshake'),'--output',str(root/'child')] and parent['child_command_sha256']==sha(root/'child.command.json'),'Actual child argv identity differs')
 require(type(parent['child_pid']) is int and parent['child_pid']>0 and parent['child_started_epoch']<=parent['child_terminal_epoch']<=parent['leaf_logs_epoch']<=parent['post_health_finished_epoch'],'Actual child/leaflog/posthealth chronology differs')
 logs={str(p.relative_to(root/'child')):sha(p) for p in sorted((root/'child').rglob('*.log'))}
 require(logs and logs==parent['leaf_log_sha256'],'Original complete terminal child log roster/SHA changed')
 require(plan['kind']=='serial' or 'engine.combined.log' in logs,'Actual native/API complete producer log missing')
 for name,digest in logs.items():
  path=root/'child'/name;require(not path.is_symlink() and path.resolve().is_relative_to(root/'child'),'Actual child log escaped')
 require(all(child['artifact_bindings'].get(name,{}).get('sha256')==digest for name,digest in logs.items()),'Child-terminal artifact and parent original logSHA differ')
 post=journal_binding(root,parent,'post');path=root/'post-model-identity.json';binding=parent['post_model_identity']
 require(Path(binding['path']).resolve()==path and sha(path)==binding['sha256'],'Actual original canonical newfull4 path/hash changed')
 identity=read(path);require(type(child['finished_epoch']) in (int,float) and math.isfinite(child['finished_epoch']) and child['finished_epoch']>0,'Actual child finish epoch invalid');require(identity['passed'] is True and len(identity['rows'])==4 and all(type(identity[k]) in (int,float) and math.isfinite(identity[k]) for k in ('started','finished')) and parent['finished_epoch']>=identity['finished']>=identity['started']>=max(post['finished_epoch'],parent['child_terminal_epoch'],child['finished_epoch'],parent['post_health_finished_epoch']),'Actual identity scan before observedpostjournal/terminal/health or incomplete')
 return {'child_pid':parent['child_pid'],'child_interpreter':interpreter,'child_command_sha256':parent['child_command_sha256'],'leaf_logs':logs,'pre_journal':journal_binding(root,parent,'pre'),'post_journal':journal_binding(root,parent,'post')}
