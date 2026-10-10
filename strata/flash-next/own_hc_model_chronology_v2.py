"""Original CPU model/owned phase terminals precede EOF/GPU/post publisher."""
import math
def require(ok,msg):
 if not ok:raise ValueError(msg)
def admit(report):
 phase=report['reference_phase'];work=phase['work'];run=report['run_command'];values=[report['started_epoch'],run['started_command_epoch'],work['first_owned_terminal_epoch']]
 require(type(work['prefix4_attempted']) is bool,'Typed actual prefix4 scope required')
 if work['prefix4_attempted']:values.append(work['prefix4_computation_terminal_epoch'])
 else:require('prefix4_computation_terminal_epoch' not in work,'Unattempted prefix4 cannot borrow terminal')
 values.extend([work['computation_terminal_epoch'],phase['model_computation_terminal_epoch'],run['completed_epoch'],run['finished_epoch'],report['GPU_terminal_epoch'],report['post_health']['finished_epoch'],report['post_journal']['finished_epoch'],report['post_full4']['started'],report['post_full4']['finished'],report['finished_epoch']]);require(all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in values),'Finite actual original-model/phase/source epochs required');require(all(a<=b for a,b in zip(values,values[1:])),'Original first/prefix/final CPU and phase terminals must precede actual EOF/GPU/new4');return {'actual_original_model_phase_EOF_GPU_publisher_order':True,'prefix4_attempted':work['prefix4_attempted']}
