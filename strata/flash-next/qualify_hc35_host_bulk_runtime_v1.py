"""Root-only fresh CPU compile/qualification and read-only bulk admission."""
import argparse,hashlib,json,os,shlex,subprocess,time
from pathlib import Path
import numpy as np
import hc35_host_bulk_fma_v1 as bulk
import qualify_hc35_host_runtime_v1 as old
import hc_f32_arithmetic35_host_v2 as scalar
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
IMAGE='sha256:39992d7072aa0557f4e3a5faf7782f83bab8fe4882e9d8fcf3e660d3de50e9e7'
sha=bulk.sha;require=bulk.require;read=lambda p:json.loads(Path(p).read_text())
def write(path,row):Path(path).write_text(json.dumps(row,indent=2,ensure_ascii=True)+'\n',encoding='ascii')
def cases():
 result=[]
 for k in (32,64,320,10240):
  for mode in ('shared','paired'):
   m=2;w=(((np.arange(m*k)*17+11)%31)-15).astype('<f4')/np.float32(16);x=(((np.arange(k if mode=='shared' else m*k)*13+7)%23)-11).astype('<f4')/np.float32(8);a=w.tobytes();b=x.tobytes();result.append({'label':'matrix_'+mode+'_'+str(k),'op':'matrixdot','K':k,'M':m,'mode':mode,'a':a,'b':b,'c':None,'expected':bulk.reference_matrixdot(k,m,mode,a,b)})
 a=np.asarray([1+2**-23,2**-126,2**-149],dtype='<f4').tobytes();b=np.asarray([1-2**-23,.5,2**24],dtype='<f4').tobytes();c=np.asarray([-1.,0.,0.],dtype='<f4').tobytes();result.append({'label':'bulk_fused_gradual','op':'fma','K':3,'M':1,'mode':'paired','a':a,'b':b,'c':c,'expected':np.asarray([-2**-46,2**-127,2**-125],dtype='<f4').tobytes()});return result
def compile_recipe(out,pid=None):
 source=HERE/'hc35_host_bulk_fma_v1.cpp';argv=['g++','-O3','-std=c++17','-mfma','-fno-fast-math','-ffp-contract=off','-frounding-math','/source/hc35_host_bulk_fma_v1.cpp','-o','/out/hc35-host-bulk'];name='b70-hc35-host-bulk-'+str(os.getpid() if pid is None else pid);binding=sha(bulk.SOURCE_PLAN)
 command=['docker','run','--name',name,'--label','b70.hc35bulk.compile='+binding,'--network','none','--user','1000:1000','--cpus','2','--memory','2g','--memory-swap','2g','--entrypoint','/bin/bash','-v',str(HERE)+':/source:ro','-v',str(out)+':/out',IMAGE,'-c','set -e\ng++ --version\nexec '+shlex.join(argv)]
 return name,argv,command
def observed_compile(obj,out,name):
 require(obj['Name']=='/'+name and obj['Config']['Image']==IMAGE and obj['Config']['Labels'].get('b70.hc35bulk.compile')==sha(bulk.SOURCE_PLAN),'Compile observed image/owner changed')
 require(obj['Image']==IMAGE and obj['Config']['Cmd']==compile_recipe(out)[2][-2:],'Compile observed image ID/shell command changed')
 h=obj['HostConfig'];require(h['NetworkMode']=='none' and h['NanoCpus']==2000000000 and h['Memory']==2*(1<<30) and h['MemorySwap']==2*(1<<30) and not h.get('Privileged') and not h.get('Devices') and not h.get('DeviceRequests'),'Compile observed CPU/network/device bounds changed')
 require(obj['Config']['User']=='1000:1000' and obj['Config']['Entrypoint']==['/bin/bash'],'Compile observed user/entrypoint changed')
 mounts=[(r['Source'],r['Destination'],r['RW'],r['Type']) for r in obj['Mounts']]
 require(sorted(mounts)==sorted([(str(HERE),'/source',False,'bind'),(str(Path(out).resolve()),'/out',True,'bind')]),'Compile observed source/output-only mounts changed')
 return obj

def execute_saved(helper,command,path):
 p=subprocess.run(command,check=True,capture_output=True,text=True,timeout=300)
 return {'command':command,'stdout':p.stdout,'stderr':p.stderr,'return_code':p.returncode,'output':old.file_binding(path)}

def bulk_command(helper,case,directory,path):
 return [str(helper),case['op'],str(case['K']),str(case['M']),case['mode'],str(directory/'a.bin'),str(directory/'b.bin'),str(directory/'c.bin') if case['c'] is not None else '-',str(path),bulk.TAG]

def validate_bulk_receipt(row,case,helper,directory,path):
 require(row['command']==bulk_command(helper,case,directory,path) and row['return_code']==0 and row['stderr']=='','Exact saved bulk command/terminal changed')
 bulk.validate_receipt(json.loads(row['stdout']),case['op'],case['K'],case['M'],case['mode'],len(case['expected']))

def build_binding(root):
 root=Path(root).resolve();r=read(root/'compile-receipt.json');helper=root/'hc35-host-bulk';require(r['passed'] is True and r['return_code']==0 and r['removed'] is True and r['state']['Running'] is False and r['state']['ExitCode']==0 and not r['state']['OOMKilled'] and not r['state'].get('Error'),'Actual fresh CPU compile/owned terminal prerequisite failed');require(r['source_binding']==bulk.source_binding() and r['source_plan_sha256']==sha(bulk.SOURCE_PLAN) and r['producer_sha256']==sha(__file__) and r['helper_sha256']==sha(helper) and r['compile_log_sha256']==sha(root/'compile.log'),'Fresh bulk compile/source/binary/log association changed')
 name,argv,command=compile_recipe(root,r['producer_pid']);observed_compile(r['observed_container'],root,name);require(r['command']==command and r['compile_argv']==argv and r['container']==name and r['GPU_touch'] is False and r['model_payload_read'] is False,'Exact CPU compile image/device/owner/argv/flags differ');require(helper.read_bytes()[:4]==b'\x7fELF','Fresh host ELF required');return {'compile_receipt_sha256':sha(root/'compile-receipt.json'),'helper_sha256':sha(helper),'compile_image':IMAGE,'compile_image_does_not_pin_host_runtime':True,'source_binding':r['source_binding']}
def qualify(root,scalar_run,scalar_build):
 root=Path(root).resolve();helper=root/'hc35-host-bulk';build=build_binding(root);scalar_helper=Path(scalar_build).resolve()/'hc35-host';qualified_scalar=old.finalized_binding(scalar_run,scalar_helper,scalar_build);out=root/'qualification';require(not out.exists(),'New bulk qualification directory required');out.mkdir();(out/'outputs').mkdir();report={'schema':1,'passed':False,'errors':[],'started_epoch':time.time(),'producer_sha256':sha(__file__),'source_binding':bulk.source_binding(),'build_binding':build,'qualified_scalar':qualified_scalar,'scalar_run':str(Path(scalar_run).resolve()),'scalar_build':str(Path(scalar_build).resolve()),'runtime_pre':old.runtime_binding(helper),'scalar':[],'bulk':[],'GPU_touch':False,'model_payload_read':False,'device_intrinsics_qualified':False,'full_model_math_qualified':False,'tolerance_gate':None}
 try:
  for case in old.fixture_cases():
   fixture=old.preserve_fixture(out,case);path=out/'outputs'/(case['label']+'.bin');command=[str(helper),case['op'],str(len(case['b'])//4),fixture['files']['a']['path'],fixture['files']['b']['path'],fixture['files']['c']['path'] if case['c'] is not None else '-',str(path),scalar.TAG];p=subprocess.run(command,check=True,capture_output=True,text=True,timeout=60);raw=path.read_bytes();require(raw==case['expected'],'New helper scalar known bytes differ');report['scalar'].append({'fixture':fixture,'command':command,'stdout':p.stdout,'stderr':p.stderr,'return_code':p.returncode,'output':old.file_binding(path),'expected_hex':case['expected'].hex(),'passed':True})
  for case in cases():
   directory=out/'bulk-fixtures'/case['label'];directory.mkdir(parents=True);bindings={}
   for key in ('a','b','c','expected'):
    if case[key] is not None:
     path=directory/(key+'.bin');path.write_bytes(case[key]);bindings[key]=old.file_binding(path)
   path=out/'outputs'/(case['label']+'.bin');execution=execute_saved(helper,bulk_command(helper,case,directory,path),path);validate_bulk_receipt(execution,case,helper,directory,path);raw=path.read_bytes();require(raw==case['expected'],'Actual bulk helper exact scalar schedule bytes differ')
   equivalence=[];n=case['K'];w=case['a'];x=case['b']
   for i in range(case['M']):
    eq=directory/('scalar-'+str(i));eq.mkdir();aa=w[i*n*4:(i+1)*n*4];bb=x if case['mode']=='shared' else x[i*n*4:(i+1)*n*4];cc=case['c'][i*n*4:(i+1)*n*4] if case['c'] is not None else None
    for key,value in [('a',aa),('b',bb),('c',cc)]:
     if value is not None:(eq/(key+'.bin')).write_bytes(value)
    target=eq/'out.bin';cmd=[str(scalar_helper),'dot' if case['op']=='matrixdot' else 'fma',str(n),str(eq/'a.bin'),str(eq/'b.bin'),str(eq/'c.bin') if cc is not None else '-',str(target),scalar.TAG];saved=execute_saved(scalar_helper,cmd,target);want=raw[i*4:(i+1)*4] if case['op']=='matrixdot' else raw[i*n*4:(i+1)*n*4];require(target.read_bytes()==want,'Qualified scalar helper differs from bulk');equivalence.append(saved)
   report['bulk'].append(dict(execution,label=case['label'],inputs=bindings,output_hex=raw.hex(),scalar_equivalence=equivalence,passed=True))

  report['runtime_post']=old.runtime_binding(helper);require(report['runtime_post']==report['runtime_pre'] and build_binding(root)==build and bulk.source_binding()==report['source_binding'],'Current CPU runtime/source/build changed');report['passed']=True
 except Exception as error:report['errors'].append(str(error))
 report['finished_epoch']=time.time();write(out/'report.json',report);return report
def finalized_binding(root):
 root=Path(root).resolve();out=root/'qualification';r=read(out/'report.json');helper=root/'hc35-host-bulk';require(r['passed'] is True and r['errors']==[] and r['producer_sha256']==sha(__file__) and r['source_binding']==bulk.source_binding() and r['build_binding']==build_binding(root),'Completed fresh bulk qualification source/build failed');scalar_helper=Path(r['scalar_build'])/'hc35-host';require(r['qualified_scalar']==old.finalized_binding(r['scalar_run'],scalar_helper,r['scalar_build']),'Qualified scalar prerequisite changed');current=old.runtime_binding(helper);require(r['runtime_pre']==r['runtime_post']==current,'Current host ELF/libm/environment/CPU/rounding binding changed');scalar_cases=old.fixture_cases();bulk_cases=cases();require(len(r['scalar'])==15 and len(r['bulk'])==9,'Exact actual15scalar+9bulk roster required')
 for row,case in zip(r['scalar'],scalar_cases):
  old.validate_fixture(out,row['fixture'],case);require(row['passed'] is True and row['return_code']==0 and row['expected_hex']==case['expected'].hex() and old.file_binding(out/'outputs'/(case['label']+'.bin'))==row['output'] and (out/'outputs'/(case['label']+'.bin')).read_bytes()==case['expected'],'Actual preserved scalar output differs')
  expected=[str(helper),case['op'],str(len(case['b'])//4),row['fixture']['files']['a']['path'],row['fixture']['files']['b']['path'],row['fixture']['files']['c']['path'] if case['c'] is not None else '-',str(out/'outputs'/(case['label']+'.bin')),scalar.TAG];require(row['command']==expected,'Actual preserved scalar argv differs');receipt=json.loads(row['stdout']);require(receipt['schema']==1 and receipt['op']==case['op'] and receipt['elements']==len(case['b'])//4 and receipt['rounding']=='FE_TONEAREST' and receipt['flush_to_zero'] is False and receipt['denormals_are_zero'] is False and receipt['device_intrinsics_qualified'] is False and receipt['model_math_qualified'] is False,'Actual scalar helper rounding/scope changed')
 for row,case in zip(r['bulk'],bulk_cases):
  require(row['label']==case['label'] and row['passed'] is True and row['output_hex']==case['expected'].hex(),'Actual bulk known schedule output differs')
  for key in ('a','b','c','expected'):
   if case[key] is not None:
    path=out/'bulk-fixtures'/case['label']/(key+'.bin');require(path.read_bytes()==case[key] and old.file_binding(path)==row['inputs'][key],'Preserved full bulk fixture changed')
  path=out/'outputs'/(case['label']+'.bin');validate_bulk_receipt(row,case,helper,out/'bulk-fixtures'/case['label'],path);require(path.read_bytes()==case['expected'] and old.file_binding(path)==row['output'],'Preserved bulk output changed')
  require(len(row['scalar_equivalence'])==case['M'],'Exact scalar equivalence roster required')
  for i,saved in enumerate(row['scalar_equivalence']):
   eq=out/'bulk-fixtures'/case['label']/('scalar-'+str(i));n=case['K'];target=eq/'out.bin';cmd=[str(scalar_helper),'dot' if case['op']=='matrixdot' else 'fma',str(n),str(eq/'a.bin'),str(eq/'b.bin'),str(eq/'c.bin') if case['c'] is not None else '-',str(target),scalar.TAG]
   for key in ('a','b','c'):
    if case[key] is not None:
     want=case[key] if key=='b' and case['mode']=='shared' else case[key][i*n*4:(i+1)*n*4];require((eq/(key+'.bin')).read_bytes()==want,'Preserved scalar equivalence operands changed')
   want=case['expected'][i*4:(i+1)*4] if case['op']=='matrixdot' else case['expected'][i*n*4:(i+1)*n*4];receipt=json.loads(saved['stdout']);require(saved['command']==cmd and saved['return_code']==0 and saved['stderr']=='' and old.file_binding(target)==saved['output'] and target.read_bytes()==want and receipt['schema']==1 and receipt['op']==cmd[1] and receipt['elements']==n and receipt['rounding']=='FE_TONEAREST' and receipt['flush_to_zero'] is False and receipt['denormals_are_zero'] is False and receipt['device_intrinsics_qualified'] is False and receipt['model_math_qualified'] is False,'Actual qualified scalar equivalence command/output/rounding differs')

 require(r['GPU_touch'] is False and r['model_payload_read'] is False and r['device_intrinsics_qualified'] is False and r['full_model_math_qualified'] is False,'Bulk qualification scope changed');return {'report_sha256':sha(out/'report.json'),'source_binding':r['source_binding'],'build_binding':r['build_binding'],'current_host_runtime':current,'scalar_cases':15,'bulk_cases':9,'device_intrinsics_qualified':False,'full_model_math_qualified':False}
def main():
 p=argparse.ArgumentParser();p.add_argument('--build-root',type=Path,required=True);p.add_argument('--scalar-run-root',type=Path,required=True);p.add_argument('--scalar-build-root',type=Path,required=True);a=p.parse_args();out=a.build_root.resolve();require(not out.exists(),'Fresh CPU build directory required');binding=bulk.source_binding();old.finalized_binding(a.scalar_run_root,a.scalar_build_root.resolve()/'hc35-host',a.scalar_build_root);require(' fma ' in ' '+Path('/proc/cpuinfo').read_text().split('flags',1)[1].split('\n',1)[0]+' ','Actual CPU FMA capability required');out.mkdir(parents=True);name,argv,command=compile_recipe(out);r={'passed':False,'errors':[],'producer_pid':os.getpid(),'source_binding':binding,'source_plan_sha256':sha(bulk.SOURCE_PLAN),'producer_sha256':sha(__file__),'command':command,'compile_argv':argv,'container':name,'GPU_touch':False,'model_payload_read':False}
 try:
  with (out/'compile.log').open('w') as log:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=300)
  r['return_code']=result.returncode;require(result.returncode==0,'Fresh CPU bulk compile failed');r['helper_sha256']=sha(out/'hc35-host-bulk');r['compile_log_sha256']=sha(out/'compile.log');require(bulk.source_binding()==binding,'Bulk source changed during compile')
 except Exception as error:r['errors'].append(str(error))
 finally:
  try:
   obj=json.loads(subprocess.check_output(['docker','inspect',name],text=True))[0];require(obj['Name']=='/'+name and obj['Config']['Image']==IMAGE and obj['Config']['Labels'].get('b70.hc35bulk.compile')==sha(bulk.SOURCE_PLAN),'Compile owner differs')
   if obj['State']['Running']:subprocess.run(['docker','stop','--time','15',name],check=True);obj=json.loads(subprocess.check_output(['docker','inspect',name],text=True))[0]
   r['observed_container']=observed_compile(obj,out,name);r['state']=obj['State'];require(not r['state']['Running'] and r['state']['ExitCode']==0 and not r['state']['OOMKilled'],'Compile terminal failed');subprocess.run(['docker','rm',name],check=True,capture_output=True);r['removed']=not subprocess.check_output(['docker','ps','-aq','--filter','name=^/'+name+'$'],text=True).strip()
  except Exception as error:r['errors'].append(str(error))
 r['passed']=not r['errors'];write(out/'compile-receipt.json',r);require(r['passed'],'CPU bulk fresh compile/teardown failed');report=qualify(out,a.scalar_run_root,a.scalar_build_root);print(json.dumps({'passed':report['passed'],'GPU_touch':False,'model_math_qualified':False}));return int(not report['passed'])
if __name__=='__main__':raise SystemExit(main())
