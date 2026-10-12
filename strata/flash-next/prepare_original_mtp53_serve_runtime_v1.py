"""Pure explicit native qualification intake; HTTP/cache adapters remain unqualified."""
import argparse,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
PLAN=HERE/'original-mtp53-serve-runtime-source-plan-v1.json'
MTP=ROOT/'models/files/qwen3.8-flash-next/unsloth-ud-q4-k-xl/MTP/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(v,s):
 if not v:raise ValueError(s)
def flag(args,key,value):
 if key in args:args[args.index(key)+1]=str(value)
 else:args.extend([key,str(value)])
def prepare(compile_root,receipt_sha,elf_sha,base_recipe,base_sha,output):
 plan=json.loads(PLAN.read_bytes())
 for p,h in plan['files'].items():require(sha(ROOT/p)==h,'Runtime intake source changed '+p)
 compile_root=Path(compile_root);output=Path(output);receipt=compile_root/'output/receipt.json';elf=compile_root/'output/original-mtp53-serve';recipe_path=compile_root/'compile-recipe.json'
 require(sha(receipt)==receipt_sha and sha(elf)==elf_sha,'Actual fresh serving compile receipt/ELF required')
 build=json.loads(receipt.read_bytes());require(build['passed']is True and build['owned_removed']is True and build['devices_exposed']is False and build['fresh_MAIN_and_header_consumer_compiled']is True and build['CPU_contract_executed']is True and build['CPU_contract_passed']is True and build['CPU_contract_controls']==7 and build['outputs']['original-mtp53-serve']==elf_sha,'Fresh MAIN+header compile required')
 require(build['source_plan_sha256']==sha(HERE/'original-mtp53-serve-builder-source-plan-v2.json')and build['recipe_sha256']==sha(recipe_path),'Actual immutable compile recipe required')
 require(sha(base_recipe)==base_sha,'Authenticated exact base software recipe required');base=json.loads(Path(base_recipe).read_bytes());require(type(base['native_argv'])is list and type(base['native_env'])is dict,'Actual native argv/environment recipe required')
 argv=list(base['native_argv']);env=dict(base['native_env']);require('--serve'in argv and '--native'in argv and '--mtp'not in argv,'Native main serving/pack/no alternate draft required')
 for key,val in [('--batch',4),('--batch-groups',1),('--layer-split',24),('--split-device',1),('--spec',2),('--max-context',4096),('--prefill',64),('--expert-cache',6500),('--pipeline-windows',0),('--adapt-every',0),('--lookup-chain',0),('--suffix-draft',0),('--short-read',0),('--conversation-cache-mib',512),('--conversation-cache-slots',8),('--prompt-cache',3),('--prompt-cache-root',1),('--adapt-async',0)]:flag(argv,key,val)
 for key in ['--trim-stage-weights','--no-prefill-borrow','--stop-eos','--no-kv-grow']:
  if key not in argv:argv.append(key)
 flag(argv,'--eos-ids','248044,248046')
 # Executable mount path is explicit in the root controller; base SDK binary is not relabelled.
 argv[0]='/original-serve/original-mtp53-serve'
 env.update(STRATA_ORIGINAL_MTP53='1',STRATA_ORIGINAL_MTP53_SERVING='1',STRATA_ORIGINAL_MTP53_MAIN_GGUF='/original.gguf',STRATA_ONE_TOKEN_COMMIT='0',STRATA_VERIFY_MIRROR_STAGE_T1='0',STRATA_COPY_ENGINE='0',STRATA_PF_GROUPED_F16='0',STRATA_PF_ESIMD_F16='0',STRATA_HOST_CRITICAL_PATH36='0',STRATA_KV_GROW='0',STRATA_BATCH_FULL_STATE_CHAIN='1',STRATA_BATCH_PUBLIC_PREFIX='1',STRATA_FULL_CACHE_OBSERVER38='0',ZE_AFFINITY_MASK='0,1')
 env['STRATA_ORIGINAL_MTP53_DRAFT_CONTRACT_SHA256']=sha(HERE/'original-mtp53-serve-source-plan-v2.json');env['STRATA_ORIGINAL_MTP53_TARGET_CONTRACT_SHA256']=build['actual58']['receipt_sha256']
 require(output.is_absolute()and output.resolve()==output and not output.exists(),'New canonical external intake required')
 value={'schema':1,'source_plan_sha256':sha(PLAN),'served_model_names':['hotschmoe-dd','qwen3.8-flash-next-unsloth-ud-q4-k-xl-original-mtp53-research'],'native_argv':argv,'native_env':env,'matched_arms':[{'name':'target-baseline','environment_override':{'STRATA_ORIGINAL_MTP53_SERVING':'0'}},{'name':'original-private-MTP','environment_override':{'STRATA_ORIGINAL_MTP53_SERVING':'1'}}],'cards':[0,1],'stages':[[0,0,24],[1,24,48]],'additional_readonly_mounts':{str(compile_root/'output'):'/original-serve',str(MTP):'/original.gguf'},'actual_compile_receipt':str(receipt),'actual_compile_receipt_sha256':receipt_sha,'actual_ELF_sha256':elf_sha,'authenticated_base_recipe_sha256':base_sha,'required_root_boundaries':['fresh full4 MAIN GGUF plus whole original MTP SHA/stat5 before/after','both-card strict and compiled collective pre/post health','kernel window','typed native EOF/Exit0/removal','fresh original math capture before API promotion'],'qualification_sequence':['four distinct RID/gen1 admissions and >=64 target picks matching closed native greedy references','reuse/cancel/stale RID and new generation; exact complete states','ORIGINAL_SAVE plus independent new process ORIGINAL_LOAD digest and divergent suffix continuation matching fresh','HTTP original-mode adapter: /v1/models stable primary alias, four concurrent coherent streams, cancellation, full compound cache persistence'],'arm_delta':'sole original serving enable flag; each uses fresh process, same actual SDK MAIN/config/input','baseline_completion_required_before_ON':True,'four_slot_reference':'baseline full output IDs, EOS and reset histories must match every original-mode stream; no prefix-only equality','primary_alias_order_required':True,'GPU_launched':False,'model_payload_read':False,'native_qualified':False,'API_qualified':False,'cache_qualified':False,'legacy_HTTP_disk_session_compatible':False,'scope':'Runnable native recipe intake only; actual API must implement ORIGINAL_SAVE/LOAD schema before persisted-cache claims'}
 output.write_text(json.dumps(value,indent=2)+'\n',encoding='ascii');return value
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for k in ['compile-root','receipt-sha256','elf-sha256','base-recipe','base-recipe-sha256','output']:p.add_argument('--'+k,required=True)
 a=p.parse_args();print(json.dumps(prepare(Path(a.compile_root),a.receipt_sha256,a.elf_sha256,Path(a.base_recipe),a.base_recipe_sha256,Path(a.output)),ensure_ascii=True))
