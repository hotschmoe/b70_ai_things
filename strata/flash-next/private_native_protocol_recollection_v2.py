"""Read-only actual native protocol/row identity reconstruction, no GPU calls.

OFF has no tensor/capture/N-row observation authority. Cancellation issuance is
reconstructed from the pinned source predicate and actual terminal ack; stdin
commands were not separately logged by the frozen stream.
"""
import json,re
from batch_numerical_protocol_v2 import Roster,tags
from batch_numerical_prefixes_v2 import serial_jobs

def require(ok,message):
 if not ok:raise ValueError(message)
def fields(line):return dict(re.findall(r'(\w+)=([^ ]+)',line))
def recollect(plan,trace,saved_requests):
 n=plan['slots'];diagnostic=plan['diagnostic'];require(type(n) is int and n in (2,4,6) and type(diagnostic) is int and diagnostic in (0,1),'Bounded actual native2/4/6 OFF/ON required')
 require(plan['max_new']==32 and plan['max_new_by_request']==[32]*n and type(plan['cancel_index']) is int and 0<=plan['cancel_index']<n,'Exact actual32/cancel budget required; no native64 transfer')
 require(set(saved_requests)=={str(i) for i in list(range(1001,1001+n))+list(range(2001,2001+n))},'Complete actual warm/target RID roster required')
 roster=Roster(n);warm=list(range(1001,1001+n));targets=list(range(2001,2001+n));cancel=targets[plan['cancel_index']]
 for index,rid in enumerate(warm):roster.submit(rid,index,plan['tokens']['warm'][index],32)
 target_registered=False;arm_index=None;arm_seen=False;cancel_sent=False;cancel_source=None;ready=False;protocol_lines=[];phase_traffic={'warm':set(),'target':set()}
 for number,line in enumerate(trace.splitlines()):
  if line.startswith('HARNESS ARM'):
   require(diagnostic==1 and not arm_seen and line=='HARNESS ARM after actual unarmed multi-row terminal','Only exact unique real ON warm-to-ARM marker allowed');require(all(roster.requests[r]['done'] for r in warm) and not roster.owners,'ARM before actual warm drain');arm_seen=True;arm_index=len(roster.events);continue
  if not diagnostic:require(not line.startswith('SBF '),'OFF cannot borrow or emit observer data/events')
  elif not arm_seen:require(not line.startswith(('SBF request ','SBF replay ','SBF vector ','SBF allocation ')),'Unarmed warm observer allocation/capture occurred')
  if line.startswith(('T ','BT ','DONE ','BADM ','BDONE ')):
   rid,_=tags(line);require(ready,'Native output before actual READY')
   if rid in targets and not target_registered:
    require(all(roster.requests[r]['done'] for r in warm) and (arm_seen or not diagnostic),'Target before exact warm/ARM boundary');roster.bt_slots.clear()
    for index,target in enumerate(targets):roster.submit(target,index,plan['tokens']['target'][index],32)
    target_registered=True
   if line.startswith('BT '):phase_traffic['target' if rid in targets else 'warm'].add(int(line.split()[1]))
   protocol_lines.append({'line_index':number,'line':line})
  roster.consume(line)
  if line.startswith('READY'):require(not ready,'Duplicate native READY');roster.capacity();ready=True
  if target_registered and not cancel_sent and roster.requests[cancel]['admitted']:
   allowed=(any(int(e['rows'])>=2 and int(e['active_mask'])&(1<<roster.requests[cancel]['slot']) for e in roster.events[arm_index:]) if diagnostic and arm_index is not None else len(roster.bt_slots)>=2 if not diagnostic else False)
   if allowed:
    command=roster.cancel(cancel);cancel_sent=True;cancel_source={'line_index':number,'predicate_trigger_line':line,'source_predicate_command':command,'actual_stdin_bytes_logged':False}
 require(ready and target_registered and cancel_sent and not roster.owners and all(roster.requests[r]['done'] for r in warm+targets),'Actual complete native warm/target/real cancel terminals required');roster.capacity();actual={str(k):v for k,v in roster.requests.items()};require(actual==saved_requests,'Reconstructed actual protocol differs from complete saved RID records')
 require(len(phase_traffic['warm'])>=2 and len(phase_traffic['target'])>=2,'Actual warm/target private-slot decode traffic absent')
 rows={};counter_rows={};resume_observed=set()
 for line in trace.splitlines():
  if line.startswith('SBF resume '):
   f=fields(line)
   if all(k in f for k in ('rid','reused','read_from','reread_to')):resume_observed.add(int(f['rid']))
 args=plan.get('args',[]);eos=[248044,248046]
 if '--eos-ids' in args:eos=[int(t) for t in args[args.index('--eos-ids')+1].split(',')]
 for phase,rids in [('warm',warm),('target',targets)]:
  for index,rid in enumerate(rids):
   row=actual[str(rid)];admission=row['admission_done'].split();require(int(admission[8])==0 and int(admission[14])==len(row['ids']),'Actual cold admission resume/read count differs');terminal=row['done'].split();finish=terminal[3] if terminal[0]=='BDONE' else terminal[5];require(terminal[0]=='BDONE' and finish in ('stop','length','cancel'),'Actual private slot terminal absent')
   if phase=='warm' or rid!=cancel:require(finish!='cancel' and row['cancel_requested'] is False,'Foreign or warm cancellation')
   else:require(finish=='cancel' and row['cancel_requested'] is True,'Actual intended cancel ack absent')
   require(finish!='stop' or row['generated'] and row['generated'][-1] in eos,'Actual stop requires declared source EOS token')
   require(finish!='length' or len(row['generated'])==32,'Actual bounded length finish requires complete32; contextcut not admitted in thiscase')
   rows[phase+'_'+str(index)]={'submitted_ids':row['ids'],'sampling':{'temperature':0,'checkpoint':False},'max_new':32,'native_generated_ids':row['generated'],'native_finish':finish,'native_cancel_completed':finish=='cancel','rid':rid,'slot':row['slot'],'slot_generation':row['slotgen']};counter_rows[str(rid)]={'actual_resume_from_DONE':int(admission[8]),'actual_read_n_from_DONE':int(admission[14]),'reread_disabled_by_exact_recipe_required':True,'read_from_reread_to_observer_fields_observed':rid in resume_observed}
 return {'histories':rows,'protocol_lines':protocol_lines,'counter_rows':counter_rows,'private_slots_with_actual_BT':{k:sorted(v) for k,v in phase_traffic.items()},'cancellation':cancel_source,'actual_OFF_Nrow_tensor_event_observed':False,'actual_observer_tensor_capture_lines_present':any(l.startswith('SBF vector ') for l in trace.splitlines()),'all_emitted_IDs_reconstructed':True,'cancellation_timing_or_latency_qualified':False,'actual_serial_jobs':serial_jobs(trace,roster) if diagnostic else None,'full_model_math_qualified':False}

def compare_histories(off,on,cancel_index):
 from batch_numerical_proofs_v40 import off_on_histories
 require(set(off['histories'])==set(on['histories']),'Matched complete warm/target cohort required')
 for key,a in off['histories'].items():
  b=on['histories'][key];require(a['rid']==b['rid'] and a['slot']==b['slot'],'Logical row source differs')
 return off_on_histories(off['histories'],on['histories'],{'target_'+str(cancel_index)})
