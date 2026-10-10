"""Narrow cacheOFF extractor/absentpin adapter to frozen source37 V40 serial runtime."""
from types import SimpleNamespace
from pathlib import Path
import os,shlex
import run_batch_serial_controls_source37_v1 as shared
import serial_prefix_qualification_v8 as protocol
from merged_numerical_protocol_v2 import MergedProtocol
from extract_serial_cacheoff_numerical_v1 import extract,profile

class CacheOffMergedProtocol(MergedProtocol):
 # Preserve one producerFD/classification/close; change only absentpin request.
 request=protocol.Protocol.request

def config(plan):
 args=list(plan['args']);args[args.index('--batch')+1]='0';env=dict(plan['env']);env.update(STRATA_BATCH_FIDELITY_DIAG='0',STRATA_FIDELITY_DIAG='1',STRATA_FIDELITY_DIAG_ACTIVATIONS='1',STRATA_FIDELITY_DIAG_ARM='/results/ARM',STRATA_FIDELITY_DIAG_DIR='/results/captures',STRATA_PREFIX_DIAG='1',STRATA_PREFIX_LIFECYCLE_DIAG='1',STRATA_PREFIX_DIAG_ARM='/results/ARM',STRATA_LAYER0_Q8_DIAG='0');profile(args,env);return args,env

def command_recipe(plan,output,group_index,pid):
 args,env=config(plan);directory=Path(output)/('serial-'+str(group_index));binding=shared.providers(plan['lane'])[0].sha(Path(output)/'plan.snapshot.json');engine=Path(plan['engine_root']);container='b70-prefix-'+str(pid)+'-serial'+str(group_index);model=shared.ROOT/shared.providers(plan['lane'])[0].read(shared.HERE/'model-lock.json')['destination'];command=['docker','run','-i','--name',container,'--label','b70.prefix.plan='+binding,'--network','none','--device','/dev/dri','--user','1000:1000','--memory','105g','--memory-swap','105g','--group-add',str(os.stat('/dev/dri/renderD128').st_gid),'-v',str(engine/'build')+':/build:ro','-v',str(engine/'source')+':/src:ro','-v',plan['pack']+':/pack:ro','-v',str(model)+':/model:ro','-v',str(directory.resolve())+':/results']
 for key,value in sorted(env.items()):command+=['-e',key+'='+str(value)]
 command+=[plan['image'],'exec 2>&1; cd /src; source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; exec '+shlex.join(['/build/strata','--serve']+args)]
 return command

def run(plan,batch_output,output,pre_health,group_index):
 config(plan)
 return shared.run(plan,batch_output,output,pre_health,group_index)
