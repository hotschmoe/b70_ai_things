"""Command-only immutable retry: identical genuine manifests, qualified C388 venv."""
import argparse,hashlib,json,os
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
PLAN=HERE/'original-mtp53-artifact-preflight-stage-source-plan-v2.json'
PYTHON='/opt/b70-c1-python/bin/python'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def need(v,s):
 if not v:raise ValueError(s)
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=True)+'\n',encoding='ascii')
def corrected_command(command,old_root,new_root):
 argv=list(command);old='exec python3 /preflight.py';need(argv[-1].count(old)==1,'Exact failed system interpreter seam required');argv[-1]=argv[-1].replace(old,'exec '+PYTHON+' /preflight.py',1)
 bind=str(old_root)+':/artifact:ro';need(argv.count(bind)==1,'Exact original artifact bind required');argv[argv.index(bind)]=str(new_root)+':/artifact:ro'
 argv[argv.index('--name')+1]='b70-mtp-artifact-preflight-v2-'+str(os.getpid());argv[argv.index('--label')+1]='b70.mtp.artifact.preflight='+str(os.getpid());return argv

def prepare(a):
 plan=read(PLAN)
 for p,h in plan['files'].items():need(sha(ROOT/p)==h,'Preflight successor source changed')
 old=a.artifacts;out=a.output;need(old.is_absolute()and old.resolve()==old and sha(old/'binding.json')==a.artifact_binding_sha256,'Exact existing V1 artifact binding required');need(out.is_absolute()and out.resolve()==out and not out.exists(),'New immutable preflight stage required')
 binding=read(old/'binding.json');recipe=read(old/'preflight-command.json');need(recipe['artifact_binding_sha256']==a.artifact_binding_sha256,'Exact original preflight command binding required')
 rt=read(a.runtime_receipt);need(sha(a.runtime_receipt)==a.runtime_receipt_sha256 and rt['passed']is True and rt['image']==binding['runtime_image']and rt['recipe_files']['Dockerfile']==sha(HERE/'c1_runtime/Dockerfile')and rt['recipe_files']['probe.py']==sha(HERE/'c1_runtime/probe.py'),'Authentic C388 venv/probe source identity required')
 inputs={old/'binding.json':a.artifact_binding_sha256,old/'preflight-command.json':sha(old/'preflight-command.json'),old/'runtime-intake.json':sha(old/'runtime-intake.json')};out.mkdir()
 for name in ('binding.json','runtime-intake.json'):(out/name).write_bytes((old/name).read_bytes())
 for r in binding['records'].values():
  for field in ('manifest','runtime'):
   path=Path(r[field]);need(sha(path)==r[field+'_sha256'],'Existing actual artifact changed');inputs[path]=r[field+'_sha256'];(out/path.name).write_bytes(path.read_bytes())
 argv=corrected_command(recipe['argv'],old,out);write(out/'preflight-command.json',{**recipe,'argv':argv,'preflight_command_generation':2,'qualified_runtime_interpreter':PYTHON,'original_preflight_command_sha256':sha(old/'preflight-command.json')})
 for p,h in inputs.items():need(sha(p)==h,'Closing original artifact bytes changed')
 receipt={'schema':2,'source_plan_sha256':sha(PLAN),'original_artifact_root':str(old),'original_artifact_binding_sha256':a.artifact_binding_sha256,'staged_binding_sha256':sha(out/'binding.json'),'artifact_manifests_unchanged':True,'native_artifact_digests_unchanged':True,'original_preflight_failure_preserved':True,'original_preflight_command_sha256':sha(old/'preflight-command.json'),'new_preflight_command_sha256':sha(out/'preflight-command.json'),'qualified_interpreter':PYTHON,'actual_runtime_receipt_sha256':a.runtime_receipt_sha256,'GPU_executed':False,'NoDri_preflight_executed':False,'native_or_API_qualified':False};write(out/'stage-receipt.json',receipt);return receipt
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--artifacts',type=Path,required=True);p.add_argument('--artifact-binding-sha256',required=True);p.add_argument('--runtime-receipt',type=Path,required=True);p.add_argument('--runtime-receipt-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(prepare(a)))
