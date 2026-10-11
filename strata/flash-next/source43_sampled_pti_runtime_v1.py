"""Fresh sampled tool intake; no old installed ELF/receipt hash transfer."""
import ast,copy
from pathlib import Path
import source43_pti_runtime_v1 as original
from serial37_canonical_json_v3 import read_unique,canonical
require=original.require;regular=original.regular;sha=original.sha;inherited_env=original.inherited_env
HERE=Path(__file__).resolve().parent
IMAGE='sha256:c388186da30785b302c9c76c0ce8ca5e5351c9783c628177f6f7b9eed2f4ad17'
SOURCE=Path('/mnt/vm_8tb/b70/build/pti-unitrace-sampled-query-source-v1')
BUILD=Path('/mnt/vm_8tb/b70/build/pti-unitrace-sampled-query-c388-build-v1')
PROPOSAL=HERE/'pti-sampled-query-tool-build-proposal-v1.json';PROPOSAL_SHA='0d8303d8214d70bc7d9d85c0db3c71a61f2ea05ea3f2fdebb990cc98cced79ee'
RESULTS=original.DEPS.parent.parent
DEPS=RESULTS/'pti-sampled-query-no-device-metadata-v1/receipt-v2.json'
PROPERTY=RESULTS/'pti-sampled-query-gpu0-clock-property-v1/device-clock-properties.json'
OPTIONS=original.OPTIONS;CLOCK_ENV={**original.CLOCK_ENV,'UNITRACE_CLOCK_QUERY_SAMPLE_512':'1'};NATIVE_ENV=original.NATIVE_ENV
BUILD_SHA='288beb7f742685d77b9925618e25c86ffaa1ebda7085128dec45c36dcd766b21'
TOOL_SHA='9f481a37a2884f7c7d499f33b5f41dbc00e2c0d089a9e786d059881803417b0c'
LIB_SHA='15c1c54b837631705904eea906d34bd20d3a78f6dc61c6f38e9ceb3acfe3f64a'
DEPS_SHA='58cb9748a4420c96cb60db89fb8f099fa0def84783012ccd705282c1fa9a6374'
PROPERTY_SHA='ca646c4245e20fb44d6bcbc5a4a214f578d8f1b44f2b1e7ae2c68f4477aac289'
QUERY_SHA='df5f5955b9434b59a4d1dfccc9a38eb54ce7034f914187710fb5d12519dc6232'

def binding(path,expected_intake_sha256):
    path=Path(path);regular(path,expected_intake_sha256,1<<20);value=read_unique(path)
    require(value['build_receipt_sha256']==BUILD_SHA and value['dependency_receipt_sha256']==DEPS_SHA and value['actual_device_clock_metadata_sha256']==PROPERTY_SHA and value['unitrace_sha256']==TOOL_SHA and value['injection_sha256']==LIB_SHA,'Exact actual fresh tool/deps/API-query identities required')
    require(value['query_sample_every']==512 and type(value['query_sample_every'])is int,'Exact fresh sampler intake required')
    regular(DEPS,DEPS_SHA,16<<20);deps=read_unique(DEPS)
    dep_root=DEPS.parent
    for name,key in(('receipt.json','original_receipt_sha256'),('command.json','original_command_sha256'),('terminal-inspection.json','original_terminal_sha256'),('absence.json','original_absence_sha256')):regular(dep_root/name,deps[key],16<<20)
    terminal=read_unique(dep_root/'terminal-inspection.json');command=read_unique(dep_root/'command.json');absence=read_unique(dep_root/'absence.json')
    require(canonical(terminal)==canonical(deps['terminal']) and terminal['Config']['Image']==IMAGE,'Original NoDri terminal/image changed')
    for key,want in(('Running',False),('Paused',False),('Restarting',False),('OOMKilled',False),('Dead',False)):
        require(type(terminal['State'][key])is bool and terminal['State'][key]is want,'Original NoDri State differs '+key)
    for key,want in(('Pid',0),('ExitCode',0)):
        require(type(terminal['State'][key])is int and terminal['State'][key]==want,'Typed NoDri terminal differs '+key)
    require(terminal['State']['Error']=='' and terminal['State']['Status']=='exited'and type(terminal['RestartCount'])is int and terminal['RestartCount']==0,'NoDri normal terminal required')
    host=terminal['HostConfig'];require(not host['Devices']and not host['DeviceRequests']and host['NetworkMode']=='none','Actual no-device network-none required')
    require(sorted(host['Binds'])==sorted(command[i+1]for i,x in enumerate(command)if x=='-v') and terminal['Config']['Entrypoint']==['/bin/bash']and canonical(terminal['Config']['Cmd'])==canonical(command[-2:]),'Original exact NoDri mounts/argv changed')
    require(type(absence['return_code'])is int and absence['return_code']!=0 and 'no such object'in absence['stderr'].lower(),'Original NoDri owned absence required')
    regular(PROPOSAL,PROPOSAL_SHA,8<<20);proposal=read_unique(PROPOSAL)
    for n,want in proposal['expected_full_source_sha256'].items():regular(SOURCE/n,want,8<<20)
    for n,target in proposal['expected_source_symlinks'].items():require((SOURCE/n).is_symlink()and str((SOURCE/n).readlink())==target,'Original declared source link differs')
    regular(BUILD/'receipt.json',value['build_receipt_sha256'],8<<20);build=read_unique(BUILD/'receipt.json')
    require(build['original_creator_terminal']is True and build['original_start_client_terminal']is True and build['errors']==[],'Fresh original build clients must retire')
    tool_sha=build['installed_files']['install/bin/unitrace']['sha256'];lib_sha=build['installed_files']['install/lib/libunitrace_tool.so']['sha256']
    require(value['unitrace_sha256']==tool_sha and value['injection_sha256']==lib_sha,'Actual installed ELF hashes must join original build roster')
    regular(PROPERTY,value['actual_device_clock_metadata_sha256'],8<<20);record=read_unique(PROPERTY);require(record['receipt_sha256']==QUERY_SHA,'Actual new API-only query receipt changed')
    namespace={**vars(original),'SOURCE':SOURCE,'BUILD':BUILD,'BUILD_SHA':value['build_receipt_sha256'],'PROPOSAL':PROPOSAL,'PROPOSAL_SHA':PROPOSAL_SHA,'TOOL_SHA':tool_sha,'LIB_SHA':lib_sha,'DEPS':DEPS,'PROPERTY':PROPERTY,'PROPERTY_SHA':value['actual_device_clock_metadata_sha256'],'QUERY_SHA':record['receipt_sha256']}
    node=copy.deepcopy(next(n for n in ast.parse(Path(original.__file__).read_bytes()).body if isinstance(n,ast.FunctionDef)and n.name=='binding'))
    # Sole actual fresh API-query script basename; all original query/build/deps
    # and raw clock/current ELF predicates execute in this private call scope.
    changed=0
    for n in ast.walk(node):
        if isinstance(n,ast.Constant)and n.value=='b70_root_pti_finalizer_gpu0_clock_property_v1.py':n.value='b70_root_pti_sampled_query_gpu0_clock_property_v1.py';changed+=1
    require(changed==1,'Exact fresh API-only source-script identity required')
    exec(compile(ast.fix_missing_locations(ast.Module([node],type_ignores=[])),__file__+'[fresh-sampled-tool-build-deps-API-query]','exec'),namespace)
    result=namespace['binding'](path);result.update(query_sample_every=512,exact_intake_sha256=expected_intake_sha256,every_instance_submit_pair_bracketed=False)
    require(sha(path)==expected_intake_sha256,'Current sampler intake changed');return result


def intake():
    regular(PROPERTY,PROPERTY_SHA,8<<20);record=read_unique(PROPERTY)
    return dict(schema=1,runtime_ready=True,source_root=str(SOURCE),install_root=str(BUILD/'install'),metadata_cap_bytes=512<<20,build_receipt_path=str(BUILD/'receipt.json'),build_receipt_sha256=BUILD_SHA,dependency_receipt_path=str(DEPS),dependency_receipt_sha256=DEPS_SHA,actual_device_clock_metadata_path=str(PROPERTY),actual_device_clock_metadata_sha256=PROPERTY_SHA,actual_device_clock_properties=original.properties(record['clock_properties']),unitrace_sha256=TOOL_SHA,injection_sha256=LIB_SHA,query_sample_every=512)
if __name__=='__main__':
    import argparse,json
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();require(a.output.is_absolute()and not a.output.exists()and a.output.parent.resolve()==a.output.parent,'New canonical runtime intake output required');a.output.write_text(json.dumps(intake(),sort_keys=True,indent=2)+'\n');result=binding(a.output,sha(a.output));print(json.dumps({'passed':True,'intake':str(a.output),'sha256':sha(a.output),'actual_fresh_tool_binding':result},sort_keys=True))
