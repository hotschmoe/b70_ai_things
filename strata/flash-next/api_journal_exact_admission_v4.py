"""Actual original kernel-journal row/error/hash/epoch admission, no field rewrite."""
import math
from pathlib import Path
from batch_numerical_proofs_v7 import read,sha,require
import qualify_batch_numerical_v7 as original

def journal_gate(root,parent):
 root=Path(root).resolve();rows=parent['kernel_journal_rows'];require(set(rows)=={'pre-kernel-journal','post-kernel-journal'} and parent['journal_admission_generation']==4,'Both actual original journal rows required')
 started=parent['started_epoch'];require(type(started) in (int,float) and math.isfinite(started) and started>0,'Actual journal start epoch invalid');command=['journalctl','-k','--since','@'+str(int(started)),'--no-pager']
 for stage in ('pre','post'):
  label=stage+'-kernel-journal';row=rows[label];path=root/(label+'.log');cmd=root/(label+'.command.json');require(Path(row['path']).resolve()==path and row['return_code']==0 and row['error'] is None and row['command']==command==read(cmd) and sha(cmd)==row['command_file_sha256'] and sha(path)==row['sha256'],'Actual original journal command/error/path/hash association differs')
  epoch=row['observed_epoch'];require(type(epoch) in (int,float) and math.isfinite(epoch) and parent['finished_epoch']>=epoch>=parent[stage+'_health_finished_epoch'],'Actual journal observed epoch/health/parent chronology differs')
  require(not original.FAULT.search(path.read_text(errors='replace')),'Actual current original journal has GPU fault signature')
 require(rows['pre-kernel-journal']['observed_epoch']<=parent['child_terminal_epoch']<=rows['post-kernel-journal']['observed_epoch'],'Actual pre/post journal rows do not bracket child terminal')
 reader=parent['child_stdout_reader'];require(reader['completed'] is True and reader['error'] is None and type(reader['eof_epoch']) in (int,float) and math.isfinite(reader['eof_epoch']) and reader['eof_epoch']<=parent['child_terminal_epoch'] and sha(root/'child-supervisor.log')==parent['child_stdout_log_sha256'],'Actual child stdout EOF/error/original log hash differs')
 identity=read(root/'post-model-identity.json');require(parent['post_model_identity']['started']==identity['started'] and parent['post_model_identity']['finished']==identity['finished'],'Copied actual identity epoch differs from original JSON')
 return {'actual_original_journal_rows':rows,'original_rows_or_errors_rewritten':False,'exact_command_hash_epoch_and_error_gate':True}
