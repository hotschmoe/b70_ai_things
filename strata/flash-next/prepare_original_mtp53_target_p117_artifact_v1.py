"""Root-run actual custom ELF artifact manifests; no arbitrary native session digest."""
import argparse,hashlib,json,os,shlex
from pathlib import Path
import prepare_original_mtp53_target_p117_runtime_v1 as runtime
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
PLAN=HERE/'original-mtp53-target-p117-artifact-source-plan-v1.json'
PYTHON_SOURCES=('serve/batch_request_identity.py','serve/server.py','serve/frontend.py','serve/artifact_identity.py','tools/strata_tokenizer.py','tools/gguf_reader.py')
TOKENIZER_FILES=('vocab.json','merges.txt','token_type.json','tokenizer.json','chat_template.jinja')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def need(v,s):
 if not v:raise ValueError(s)
def write(p,v):Path(p).write_text(json.dumps(v,sort_keys=True,indent=2,ensure_ascii=True)+'\n',encoding='ascii')
def modes(native_runtime):
 rows={}
 for mode in (0,1):
  env=dict(native_runtime['native_env']);env.update(STRATA_ORIGINAL_MTP53_SERVING=str(mode),STRATA_CRITICAL_PATH_TRACE='0',STRATA_BATCH_FIDELITY_DIAG='0',STRATA_FIDELITY_DIAG='0',STRATA_LAYER0_Q8_DIAG='0',STRATA_PREFIX30='0',STRATA_PLE_INPUT33='0',SYCL_UR_TRACE='0',STRATA_SLOT_OWNER_TRACE='1',STRATA_MIRROR_OWNER_TRACE='1');env.pop('STRATA_ARTIFACT_IDENTITY_SHA256',None)
  rows[str(mode)]={'exe':'/original-serve/original-mtp53-serve','args':native_runtime['native_argv'][1:],'env':env}
 return rows

def prepare(a):
 plan=read(PLAN)
 for name,h in plan['files'].items():need(sha(ROOT/name)==h,'Artifact source changed '+name)
 out=a.output;need(out.is_absolute()and out.resolve()==out and not out.exists(),'New canonical artifact root required');out.mkdir()
 need(sha(a.base_blueprint)==a.base_blueprint_sha256 and sha(a.runtime_receipt)==a.runtime_receipt_sha256 and sha(a.pack_receipt)==a.pack_receipt_sha256,'Actual original software/runtime/pack receipts required')
 base=read(a.base_blueprint);rt=read(a.runtime_receipt);need(rt['passed']is True and rt['image']==base['image']and rt['image']=='sha256:c388186da30785b302c9c76c0ce8ca5e5351c9783c628177f6f7b9eed2f4ad17','Actual pinned C388 Python runtime required')
 intake=runtime.prepare(a.compile_root,a.build_receipt_sha256,a.elf_sha256,a.base_blueprint,a.base_blueprint_sha256,out/'runtime-intake.json');build=read(a.compile_root/'output/receipt.json');sdk=Path(build['actual58']['SDK']);pack=Path(next(s for s,d in base['read_only_mounts'].items()if d=='/pack'));pack_receipt=read(a.pack_receipt)
 for rel,row in pack_receipt['RESULT']['files'].items():need(sha(pack/rel)==row['sha256'],'Actual native pack changed '+rel)
 lock=read(HERE/'model-lock.json');wanted=[r for r in lock['files']if r['path'].startswith('UD-Q4_K_XL/')];need(len(wanted)==4,'Exact four original publisher shards required')
 profiles=[(s,d)for s,d in base['read_only_mounts'].items()if d=='/profiles/expert-profile.bin'];need(len(profiles)==1,'Actual consumed profile mount required')
 allbindings={str(a.compile_root/'output/receipt.json'):a.build_receipt_sha256,str(a.compile_root/'compile-recipe.json'):build['recipe_sha256'],str(a.runtime_receipt):a.runtime_receipt_sha256,str(a.pack_receipt):a.pack_receipt_sha256,str(a.base_blueprint):a.base_blueprint_sha256,str(HERE/'model-lock.json'):sha(HERE/'model-lock.json'),str(Path(profiles[0][0])):sha(profiles[0][0])}
 for name,h in build['outputs'].items():allbindings[str(a.compile_root/'output'/name)]=h
 allbindings[str(a.compile_root/'output/libstrata_engine.without_verify.a')]=build['filtered_engine_sha256']
 for name,h in build['original_inputs_sha256'].items():allbindings[name]=h
 for rel,row in pack_receipt['RESULT']['files'].items():allbindings[str(pack/rel)]=row['sha256']
 for file in TOKENIZER_FILES:allbindings[str(pack/'tokenizer'/file)]=sha(pack/'tokenizer'/file)
 sources={name:sha(sdk/'source'/name)for name in PYTHON_SOURCES};configs=modes(intake);records={}
 for mode,cfg in configs.items():
  env={k:str(v)for k,v in sorted(cfg['env'].items())if k.startswith(('STRATA_','SYCL_','ONEAPI_'))and k!='STRATA_ARTIFACT_IDENTITY_SHA256'}
  manifest={'schema':1,'primary_model_name':'hotschmoe-dd','research_alias':'qwen3.8-flash-next-Unsloth-UD-Q4_K_XL-original-MTP53-native58-custom-P117-observer-mode'+mode,'model':{'repo':lock['repo'],'revision':lock['revision'],'artifact':'UD-Q4_K_XL','tokenizer_gguf':'/model/'+wanted[0]['path'],'tokenizer_gguf_sha256':wanted[0]['sha256']},'tokenizer_files':{n:sha(pack/'tokenizer'/n)for n in TOKENIZER_FILES},'runtime':{'exe_sha256':a.elf_sha256,'args':cfg['args'],'env':env,'python_versions':rt['python_versions'],'python_sources':sources,'expert_profile_sha256':sha(profiles[0][0]),'original_native_binding':{'actual58':build['actual58'],'custom_build_receipt_sha256':a.build_receipt_sha256,'custom_compile_recipe_sha256':build['recipe_sha256'],'custom_MAIN_verify_outputs':build['outputs'],'copied_engine_without_old_verify_sha256':build['filtered_engine_sha256'],'original_native_pack_receipt_sha256':a.pack_receipt_sha256,'original_publisher_MTP_sha256':'8087dbb39fc73f79ac069b1debe4069a44b73ada900c8bd761569d43a5a90231','source_plan_sha256':sha(PLAN)}}}
  path=out/('artifact-mode'+mode+'.json');write(path,manifest);cfg['env']['STRATA_ARTIFACT_IDENTITY_SHA256']=sha(path);write(out/('runtime-mode'+mode+'.json'),cfg);records[mode]={'manifest':str(path),'manifest_sha256':sha(path),'runtime':str(out/('runtime-mode'+mode+'.json')),'runtime_sha256':sha(out/('runtime-mode'+mode+'.json'))}
 write(out/'binding.json',{'schema':1,'source_plan_sha256':sha(PLAN),'actual_compile_receipt_sha256':a.build_receipt_sha256,'actual_ELF_sha256':a.elf_sha256,'records':records,'host_bindings':allbindings,'sdk_source':str(sdk/'source'),'pack':str(pack),'runtime_image':rt['image'],'models_whole_payload_read_by_preparer':False,'HTTP_identity_load_pending':True,'native_runtime_preflight_required':True})
 binding_sha=sha(out/'binding.json');command=['docker','run','--name','b70-mtp-artifact-preflight-'+str(os.getpid()),'--label','b70.mtp.artifact.preflight='+str(os.getpid()),'--network','none','--cpus','2','--memory','2g','--memory-swap','2g','--user',str(os.getuid())+':'+str(os.getgid()),'-v',str(out)+':/artifact:ro','-v',str(a.compile_root/'output')+':/original-serve:ro','-v',str(sdk/'source')+':/src:ro','-v',str(pack)+':/pack:ro','-v',str(HERE/'preflight_original_mtp53_artifact_inside_v1.py')+':/preflight.py:ro','--entrypoint','/bin/bash',rt['image'],'-lc','test ! -e /dev/dri; exec /opt/b70-c1-python/bin/python /preflight.py --artifacts /artifact --binding-sha256 '+binding_sha]
 write(out/'preflight-command.json',{'argv':command,'artifact_binding_sha256':binding_sha,'devices_exposed':False,'whole_GGUF_read':False,'root_must_record_actual_terminal_EOF_removal_and_stdout':True})
 return read(out/'binding.json')

def admit(root,expected_binding_sha,native_runtime):
 plan=read(PLAN)
 for path,h in plan['files'].items():need(sha(ROOT/path)==h,'Current artifact source changed')
 root=Path(root);need(root.resolve()==root and sha(root/'binding.json')==expected_binding_sha,'Pinned actual artifact binding required');b=read(root/'binding.json');need(b['source_plan_sha256']==sha(PLAN),'Current artifact source required')
 for path,h in b['host_bindings'].items():need(sha(path)==h,'Actual artifact dependency changed '+path)
 expected=modes(native_runtime)
 for mode,r in b['records'].items():
  need(sha(r['manifest'])==r['manifest_sha256']and sha(r['runtime'])==r['runtime_sha256'],'Actual artifact/runtime changed');m=read(r['manifest']);cfg=read(r['runtime']);plain=dict(cfg['env']);digest=plain.pop('STRATA_ARTIFACT_IDENTITY_SHA256');need(digest==r['manifest_sha256']and {'exe':cfg['exe'],'args':cfg['args'],'env':plain}==expected[mode],'Actual consumed native args/environment differ from artifact');need(m['runtime']['exe_sha256']==b['actual_ELF_sha256']and m['runtime']['args']==cfg['args']and m['runtime']['env']=={k:str(v)for k,v in sorted(plain.items())if k.startswith(('STRATA_','SYCL_','ONEAPI_'))},'Actual schema runtime mismatch')
 return b
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--compile-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--build-receipt-sha256',required=True);p.add_argument('--elf-sha256',required=True)
 for key in('base-blueprint','runtime-receipt','pack-receipt'):p.add_argument('--'+key,type=Path,required=True);p.add_argument('--'+key+'-sha256',required=True)
 a=p.parse_args();v=prepare(a);print(json.dumps({'binding':str(a.output/'binding.json'),'sha256':sha(a.output/'binding.json')}))
