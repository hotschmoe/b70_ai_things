"""Shared producer/reader original command, ACK and publisher boundaries."""
import math
from pathlib import Path
from serial37_canonical_json_v3 import matches_saved,canonical
import native_qsa40_runtime_v4 as r
from native_rms_kernel_journal_v5 import reject_faults
HEALTH='sha256:d55637b3353eaf470677627dc1627c3dda3ec6a6abb0298451aed34b73937067';require=r.require

def finite(values):require(all(type(v)in(int,float)and math.isfinite(v)and v>0 for v in values),'Finite actual chronology required')
def command(out,label,row,argv):
 require(matches_saved(row,out/(label+'.receipt.json'))and r.read(out/(label+'.command.json'))==argv==row['command']and row['command_sha256']==r.sha(out/(label+'.command.json')),'Original exact typed command/receipt differs');require(row['passed']is True and type(row['return_code'])is int and row['return_code']==0 and row['error']is None and row['command_error']is None and row['phase_cleanup_error']is None and row['eof']is True and row['reader_retired']is True and row['path']==str(out/(label+'.log'))and row['sha256']==r.sha(out/(label+'.log')),'Actual command failure/EOF/log contradiction');finite([row['started_command_epoch'],row['started_epoch'],row['completed_epoch'],row['finished_epoch']]);require(row['started_command_epoch']<=row['started_epoch']<=row['completed_epoch']<=row['finished_epoch'],'Actual command EOF chronology differs')
 return row

def health(out,label,h,boundary):
 require(h['passed']is True and h['image']==HEALTH and len(h['rows'])==2,'Exact strict/compiled health roster/image required');finite([h['finished_epoch'],boundary]);last=boundary
 for row,name,argv in zip(h['rows'],('strict-health','compiled-health'),([str(r.ROOT/'vllm/int4/diagnostics/xpu_health_strict.sh'),'--img',HEALTH],[str(r.ROOT/'bin/xpu-collective-health'),'--img',HEALTH,'--p2p','0','--timeout','180'])):
  command(out,label+'-'+name,row,argv);require(last<=row['started_command_epoch']<=row['finished_epoch']<=h['finished_epoch'],'Actual health order differs');last=row['finished_epoch']
 return h

def journal(out,label,row,parent_start,after):
 command(out,label+'-kernel',row,['journalctl','-k','--since','@'+str(int(parent_start)),'--no-pager']);require(row['started_command_epoch']>=after,'Actual kernel journal before completehealth');reject_faults((out/(label+'-kernel.log')).read_text());return row

def phase(out,label,plan,ph,parent_start):
 directory=out/label;require(matches_saved(ph,out/(label+'-child.receipt.json')),'Original child phase receipt differs');ready=r.read(directory/'READY.json');ack=r.read(directory/'ACK.json');seal=r.read(directory/'GPU-seal.json');require(canonical(ready)==canonical(ph['ready'])and ready['pid']['pid']==ph['child_pid']and ready['parent']['pid']>0 and ready['plan_sha256']==r.sha(out/'plan.snapshot.json')and ready['source_plan_sha256']==r.closure(),'Actual child READY owner/plan/source differs');finite([ph['child_started_epoch'],ready['manifest_completed_epoch'],ack['ack_epoch'],seal['finished_epoch'],ph['child_terminal_epoch']]);require(ph['child_started_epoch']<=ready['manifest_completed_epoch'],'Semantic READY before actual child launch')
 health(out,label+'-pre',ph['pre_health'],ready['manifest_completed_epoch']);journal(out,label+'-pre',ph['pre_journal'],parent_start,ph['pre_health']['finished_epoch']);require(matches_saved(ph['pre_health'],out/(label+'-pre-health.json'))and ack['health_sha256']==r.sha(out/(label+'-pre-health.json'))and ack['journal_sha256']==ph['pre_journal']['sha256']and ack['ready_sha256']==r.sha(directory/'READY.json')and ack['plan_sha256']==ready['plan_sha256']and ack['health_finished_epoch']==ph['pre_health']['finished_epoch'],'Actual healthACK source/READY association differs');require(ph['pre_journal']['finished_epoch']<=ack['ack_epoch']<=seal['finished_epoch']and seal['finished_epoch']-ph['pre_health']['finished_epoch']<=300 and seal['plan_sha256']==ready['plan_sha256']and seal['manifest_current']is True,'Actual fresh source seal/healthACK chronology differs')
 from native_qsa40_byte_epoch_v4 import binding as byte_binding
 byte_binding(directory,plan,ph['child_pid'],ready,seal);byte_epoch=r.read(directory/'byte-epoch.json');require(ack['ack_epoch']<=byte_epoch['boundaries'][2]['started_epoch']and byte_epoch['boundaries'][3]['finished_epoch']<=ph['child_terminal_epoch'],'Actual byte predevice/postoperation must follow ACK and precede child retirement')
 result=r.read(directory/'result.json');cap=result['stdout_capture'];require(seal['finished_epoch']<=cap['process_started_epoch']<=cap['finished_epoch']<=ph['child_terminal_epoch'],'Actual source seal/engine EOF/child terminal order differs');supervisor=ph['stdout_capture'];require(type(ph['return_code'])is int and ph['return_code']==0 and ph['forced_cleanup']is False and ph['error']is None and supervisor['passed']is True and supervisor['eof']is True and supervisor['reader_retired']is True and supervisor['error']is None and supervisor['path']==str(out/(label+'-child.log'))and supervisor['sha256']==r.sha(out/(label+'-child.log')),'Actual child supervisor rc/EOF/log differs');finite([supervisor['started_epoch'],supervisor['completed_epoch'],supervisor['finished_epoch']]);require(ph['child_started_epoch']<=supervisor['started_epoch']<=supervisor['completed_epoch']<=supervisor['finished_epoch']<=ph['child_terminal_epoch'],'Actual child supervisor retirement differs');require(r.read(out/(label+'-child.command.json'))==ph['command']and ph['command_sha256']==r.sha(out/(label+'-child.command.json')),'Actual child CLI differs')
 return True


def parent_identity(report,ready):
 identity=report['producer_identity'];require(set(identity)=={'pid','start_ticks'}and type(identity['pid'])is int and identity['pid']>0 and type(identity['start_ticks'])is int and identity['start_ticks']>=0 and identity['pid']==report['producer_pid']and canonical(ready['parent'])==canonical(identity),'Exact original READY parent PID/start identity differs');return True
