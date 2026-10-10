"""Exact admitted serial Docker recipe, typed terminal and full49 group proof."""
import hashlib,json,os,shlex
from pathlib import Path
from serial37_canonical_json_v3 import canonical,read_unique
from batch51_serial_jobs_v1 import recollect,group
from batch_numerical_proofs_v50 import source_observers_off
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def require(ok,message):
 if not ok:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def recipe(plan,output,pid,index,gid):
 require(type(pid)is int and pid>0 and type(index)is int and index>=0 and type(gid)is int and gid>=0,'Exact actual serial PID/group/GID required');source_observers_off(plan['env']);args=list(plan['args']);require(args.count('--batch')==1,'Exact original serial batch override required');args[args.index('--batch')+1]='0';env=dict(plan['env']);env.update(STRATA_BATCH_FIDELITY_DIAG='0',STRATA_FIDELITY_DIAG='1',STRATA_FIDELITY_DIAG_ACTIVATIONS='1',STRATA_FIDELITY_DIAG_ARM='/results/ARM',STRATA_FIDELITY_DIAG_DIR='/results/captures',STRATA_PREFIX_DIAG='1',STRATA_PREFIX_LIFECYCLE_DIAG='1',STRATA_PREFIX_DIAG_ARM='/results/ARM',STRATA_LAYER0_Q8_DIAG='0');engine=Path(plan['engine_root']);model=ROOT/read_unique(HERE/'model-lock.json')['destination'];output=Path(output).resolve();directory=output/('serial-'+str(index));binding=sha(output/'plan.snapshot.json');container='b70-prefix-'+str(pid)+'-serial'+str(index)
 command=['docker','run','-i','--name',container,'--label','b70.prefix.plan='+binding,'--network','none','--device','/dev/dri','--user','1000:1000','--memory','105g','--memory-swap','105g','--group-add',str(gid),'-v',str(engine/'build')+':/build:ro','-v',str(engine/'source')+':/src:ro','-v',plan['pack']+':/pack:ro','-v',str(model)+':/model:ro','-v',str(directory)+':/results']
 for key,value in sorted(env.items()):command.extend(['-e',key+'='+str(value)])
 command.extend([plan['image'],'exec 2>&1; cd /src; source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; exec '+shlex.join(['/build/strata','--serve']+args)]);return command,args,env

def terminal(value):
 require(value['passed']is True and type(value['engine_rc'])is int and value['engine_rc']==0 and value['removed']is True and value['error']is None and value.get('failure_traceback')is None,'Actual serial group clean rc/removal/error required');state=value['state'];require(state['Running']is False and type(state['ExitCode'])is int and state['ExitCode']==0 and state['OOMKilled']is False and state['Error']=='' and state['Status']=='exited' and state['Paused']is False and state['Restarting']is False and state['Dead']is False and type(state['Pid'])is int and state['Pid']==0,'Actual typed serial Docker terminal required');return True

def binding(root,parent,plan,child):
 root=Path(root).resolve();output=root/'child';index=plan['group_index'];require(type(index)is int,'Exact group index required');directory=output/('serial-'+str(index));require(directory.is_dir() and not directory.is_symlink(),'Exact current group directory required');require(sorted(p.name for p in output.glob('serial-*')if p.is_dir())==[directory.name],'Unexpected duplicate/extra serial group directory')
 selected=group(recollect(Path(plan['batch_parent'])/'child',plan['slots']),plan['slots'],index);require(canonical(selected)==canonical(plan['actual_serial_group_binding']),'Actual collector selected group changed');require(child['passed']is True and child['collection_and_teardown_passed']is True and child['full_model_math_qualified']is False and child['plan_sha256']==parent['plan_sha256']==sha(output/'plan.snapshot.json') and type(child['actual_serial_job_count'])is int and child['actual_serial_job_count']==selected['selected_job_count'] and type(child['actual_matched_full49_vector_pairs'])is int and child['actual_matched_full49_vector_pairs']==selected['selected_full49_pairs'],'Exact child serial pass/counts/plan binding required')
 result=read_unique(directory/'result.json');terminal(result);expected,args,env=recipe(plan,output,parent['child_pid'],index,os.stat('/dev/dri/renderD128').st_gid);command=read_unique(directory/'command.json');require(canonical(command)==canonical(expected),'Actual complete serial Docker command differs from admitted plan/PID/group recipe')
 requests=read_unique(directory/'requests.json');require(len(requests)==len(selected['selected_jobs']) and canonical([r['job']for r in requests])==canonical(selected['selected_jobs']),'Actual exact selected request order/roles changed')
 for row in requests:
  path=directory/('serial-'+str(row['job']['rid'])+'-'+row['job']['role']+'.json');require(read_unique(path)==row['raw'],'Individual actual raw request differs from collected request row')
 compare=read_unique(output/'serial-comparison.json');require(compare['passed']is True and type(compare['group'])is int and compare['group']==index and type(compare['groups_required'])is int and compare['groups_required']==selected['groups_required'] and len(compare['comparisons'])==selected['selected_full49_pairs'] and canonical(compare['comparisons'])==canonical(child['comparisons']),'Complete exact group comparisons/counts differ')
 return {'result_sha256':sha(directory/'result.json'),'command_sha256':sha(directory/'command.json'),'selected_group':selected,'args':args,'env':env,'actual_normal_group_terminal':True,'full_model_math_qualified':False}
