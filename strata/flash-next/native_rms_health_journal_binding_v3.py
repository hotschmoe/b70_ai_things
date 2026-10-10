"""Exact strict/compiled P2P0 and since-floor journal argv, actual epochs."""
import math
from pathlib import Path

def require(ok,msg):
 if not ok:raise ValueError(msg)
def admit(report,repository_root,health_image):
 root=Path(repository_root);epochs=[report['started_epoch'],report['GPU_terminal_epoch']];require(all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in epochs),'Finite parent/GPU epochs required')
 expected=[[str(root/'vllm/int4/diagnostics/xpu_health_strict.sh'),'--img',health_image],[str(root/'bin/xpu-collective-health'),'--img',health_image,'--p2p','0','--timeout','180']]
 for stage in ('pre','post'):
  h=report[stage+'_health'];require(h['passed'] is True and h['image']==health_image and len(h['rows'])==2,'Exact actual percard+compiled health image/roster required');require(type(h['finished_epoch']) in (int,float) and math.isfinite(h['finished_epoch']),'Finite aggregate health epoch required')
  for row,argv in zip(h['rows'],expected):
   require(row['command']==argv and row['return_code']==0 and row['error'] is None and row['command_error'] is None and row['eof'] is True and row['reader_retired'] is True and row['passed'] is True,'Actual exact health argv/EOF/terminal differs');start=row['started_command_epoch'];finish=row['finished_epoch'];require(all(type(v) in (int,float) and math.isfinite(v) for v in (start,finish)) and (report['started_epoch'] if stage=='pre' else report['GPU_terminal_epoch'])<=start<=finish<=h['finished_epoch'],'Actual health aggregate/row/terminal chronology differs')
  require(h['rows'][0]['finished_epoch']<=h['rows'][1]['started_command_epoch'],'Actual sequential strict/compiled health order differs')
  j=report[stage+'_journal'];argv=['journalctl','-k','--since','@'+str(int(report['started_epoch'])),'--no-pager'];require(j['command']==argv and j['return_code']==0 and j['error'] is None and j['command_error'] is None and j['passed'] is True and j['eof'] is True and j['reader_retired'] is True,'Exact original journal since/argv/EOF differs');require(all(type(j[k]) in (int,float) and math.isfinite(j[k]) for k in ('started_command_epoch','finished_epoch')) and h['finished_epoch']<=j['started_command_epoch']<=j['finished_epoch'],'Actual journal must follow complete matching health')
 require(report['pre_journal']['finished_epoch']<=report['compile_command']['started_command_epoch'],'Actual prejournal must precede compile/device leaf phase');return {'actual_strict_compiled_P2P0_journal_argv_qualified':True,'actual_aggregate_health_epoch_qualified':True}
