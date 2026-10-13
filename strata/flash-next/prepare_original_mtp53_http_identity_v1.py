"""Root-only HTTP identity preparation; no model/API qualification or guard bypass."""
import argparse,hashlib,json,os
from pathlib import Path
import prepare_original_mtp53_artifact_identity_v1 as native_artifact
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];PLAN=HERE/'original-mtp53-http-identity-source-plan-v1.json'
PYTHON='/opt/b70-c1-python/bin/python'
def sha(p):return native_artifact.sha(p)
def read(p):return json.loads(Path(p).read_bytes())
def need(v,s):
 if not v:raise ValueError(s)
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=True)+'\n',encoding='ascii')
def prepare(a):
 plan=read(PLAN)
 for name,want in plan['files'].items():need(sha(ROOT/name)==want,'HTTP identity source changed '+name)
 out=a.output;need(out.is_absolute()and out.resolve()==out and not out.exists(),'New canonical HTTP identity output required')
 overlay=a.http_source;receipt=read(overlay/'receipt.json');need(overlay.resolve()==overlay and sha(overlay/'receipt.json')==a.http_source_receipt_sha256 and receipt['source_plan_sha256']==sha(HERE/'original-mtp53-http-compound-source-plan-v1.json'),'Actual prepared exact58 HTTP overlay required')
 need({str(p.relative_to(overlay)):sha(p)for p in (overlay/'serve').rglob('*')if p.is_file()}==receipt['files'],'Complete prepared HTTP package bytes required')
 rt=read(a.native_artifacts/'runtime-intake.json');binding=native_artifact.admit(a.native_artifacts,a.native_binding_sha256,rt);record=binding['records']['1'];old=read(record['manifest']);runtime=read(record['runtime']);pre=read(a.native_preflight)
 need(sha(a.native_preflight)==a.native_preflight_sha256 and pre['passed']is True and pre['artifact_binding_sha256']==a.native_binding_sha256 and pre['owned_removed']is True and pre['original_EOF']is True and type(pre['return_code'])is int and pre['return_code']==0,'Authentic closed native artifact preflight required')
 need(rt['native_env']['STRATA_ORIGINAL_MTP53_SERVING']=='1' and 'STRATA_ORIGINAL_MTP53_TARGET_P117_DIR'not in rt['native_env'],'Actual serving V5 artifact without diagnostic observer required')
 sdk_source=Path(binding['sdk_source']);need(sha(sdk_source/'serve/server.py')==read(HERE/'original-mtp53-http-compound-source-plan-v1.json')['actual58_server_sha256'],'Actual58 HTTP source required')
 sources=dict(old['runtime']['python_sources'])
 for rel in receipt['files']:
  if rel in sources or rel=='serve/original_mtp53_http_compound.py':sources[rel]=receipt['files'][rel]
 need(set(sources)==set(old['runtime']['python_sources'])|{'serve/original_mtp53_http_compound.py'},'Exact seven Python identity sources required')
 env=dict(runtime['env']);env.pop('STRATA_ARTIFACT_IDENTITY_SHA256',None)
 manifest=json.loads(json.dumps(old));manifest['research_alias']='qwen3.8-flash-next-Unsloth-UD-Q4_K_XL-original-MTP53-native58-custom-V5-HTTP-compound';manifest['runtime']['python_sources']=sources
 manifest['runtime']['env']={k:str(v)for k,v in sorted(env.items())if k.startswith(('STRATA_','SYCL_','ONEAPI_'))};manifest['runtime']['original_native_binding']['HTTP_overlay_receipt_sha256']=a.http_source_receipt_sha256;manifest['runtime']['original_native_binding']['HTTP_identity_source_plan_sha256']=sha(PLAN)
 out.mkdir();write(out/'manifest.json',manifest)
 cfg={'exe':runtime['exe'],'args':runtime['args'],'cwd':'/src','tokenizer':'/pack/tokenizer','model_name':'hotschmoe-dd','aliases':[manifest['research_alias']],'env':env,'gpu':'0,1','layer_split':'24','backend':'sycl','artifact_identity_manifest':'/http/manifest.json','slot_save_path':'/results/sessions','host':'127.0.0.1','open_browser':False,'sampling':{'temperature':0}}
 write(out/'config.json',cfg)
 # Actual parent source at /src and its changed serve package are separate RO binds.
 mounts={str(sdk_source):'/src',str(overlay/'serve'):'/src/serve',str(binding['pack']):'/pack',str(out):'/http',str(HERE/'preflight_original_mtp53_http_identity_v1.py'):'/preflight.py'}
 mounts.update(rt['additional_readonly_mounts'])
 base=read(a.base_blueprint);need(sha(a.base_blueprint)==rt['authenticated_base_recipe_sha256'],'Original authenticated native mount recipe required')
 for source,dest in base['read_only_mounts'].items():
  if dest not in('/src','/diagnostic','/original-mtp','/original-serve','/original.gguf'):mounts[source]=dest
 tokenizer=manifest['model']['tokenizer_gguf'];need(tokenizer.startswith('/model/'),'Original tokenizer GGUF mount required')
 cmd=['docker','run','--name','b70-mtp-http-identity-'+str(os.getpid()),'--label','b70.mtp.http.identity='+str(os.getpid()),'--network','none','--cpus','2','--memory','2g','--memory-swap','2g','--user',str(os.getuid())+':'+str(os.getgid())]
 for source,dest in mounts.items():cmd+=['-v',source+':'+dest+':ro']
 script='if [[ ${SETVARS_COMPLETED:-0} != 1 ]]; then source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; fi; exec '+PYTHON+' /preflight.py --config /http/config.json --manifest-sha256 '+sha(out/'manifest.json')
 cmd+=['--entrypoint','/bin/bash',binding['runtime_image'],'-lc',script]
 result={'schema':1,'source_plan_sha256':sha(PLAN),'native_artifact_binding_sha256':a.native_binding_sha256,'native_preflight_sha256':a.native_preflight_sha256,'HTTP_overlay_receipt_sha256':a.http_source_receipt_sha256,'manifest_sha256':sha(out/'manifest.json'),'config_sha256':sha(out/'config.json'),'preflight_command':cmd,'NoDri_required':True,'preflight_reads_original_first_GGUF_whole_hash':True,'preflight_starts_native_engine':False,'preflight_EXECUTED':False,'API_qualified':False,'cache_qualified':False,'launch_after_cache_and_math_qualification':['/opt/b70-c1-python/bin/python','-m','serve.server','--engine','strata','--config','/http/config.json','--host','0.0.0.0','--port','8095'],'readonly_mounts':mounts,'required_RW_results_mount':'owned-new-directory:/results:rw','required_workdir':'/src','required_GPU_visibility':'ZE_AFFINITY_MASK=0,1','required_actual_lease':'both cards','serve_launch_setvars_required':True,'serve_launch_EXECUTED':False}
 write(out/'receipt.json',result);return result
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for name in ('http-source','native-artifacts','native-preflight','base-blueprint','output'):p.add_argument('--'+name,type=Path,required=True)
 for name in ('http-source-receipt-sha256','native-binding-sha256','native-preflight-sha256'):p.add_argument('--'+name,required=True)
 print(json.dumps(prepare(p.parse_args()),ensure_ascii=True))
