"""Pure actual58 first-row target-only recipe intake; no original draft/cache authority."""
import argparse,json,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];PLAN=HERE/'prefill-first-gdn-runtime-source-plan-v1.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def flag(args,k,v):
 if k in args:args[args.index(k)+1]=str(v)
 else:args.extend([k,str(v)])
def prepare(compile_root,receipt_sha,elf_sha,base_recipe,base_sha,output):
 plan=json.loads(PLAN.read_bytes())
 for n,w in plan['files'].items():
  if sha(ROOT/n)!=w:raise ValueError('First GDN runtime source changed '+n)
 compile_root=Path(compile_root);receipt=compile_root/'output/receipt.json';recipe=compile_root/'compile-recipe.json';elf=compile_root/'output/first-gdn-target'
 if sha(receipt)!=receipt_sha or sha(elf)!=elf_sha:raise ValueError('Actual fresh first GDN build/ELF required')
 b=json.loads(receipt.read_bytes());p=json.loads(recipe.read_bytes())
 if b['passed']is not True or b['owned_removed']is not True or b['devices_exposed']is not False or b['source_plan_sha256']!=sha(HERE/'prefill-first-gdn-builder-source-plan-v1.json')or b['serve_compile_source_plan_sha256']!=sha(HERE/'prefill-first-gdn-observer-source-plan-v1.json')or b['recipe_sha256']!=sha(recipe)or b['outputs']['first-gdn-target']!=elf_sha or b['archive_removal_verified']is not True or b['fresh_MAIN_and_header_consumer_compiled']is not True:raise ValueError('Exact new observer compile/source/archive closure required')
 for n,w in b['outputs'].items():
  if sha(compile_root/'output'/n)!=w:raise ValueError('Fresh compiled object changed')
 if sha(compile_root/'output/libstrata_prefill.without_prefill.a')!=b['filtered_prefill_sha256']:raise ValueError('Actual filtered Prefill archive changed')
 if p['actual58']!=b['actual58']:raise ValueError('Actual58 dependency identity changed')
 if sha(base_recipe)!=base_sha:raise ValueError('Authentic normal CLI blueprint required')
 base=json.loads(Path(base_recipe).read_bytes());args=list(base['native_argv']);env=dict(base['native_env'])
 if '--serve'not in args or '--native'not in args or '--mtp'in args:raise ValueError('Native stock target/no alternate draft required')
 for k,v in [('--batch',4),('--batch-groups',1),('--layer-split',24),('--split-device',1),('--spec',2),('--max-context',4096),('--prefill',64),('--expert-cache',6500),('--pipeline-windows',0),('--short-read',0),('--lookup-chain',0),('--suffix-draft',0),('--adapt-every',0),('--adapt-async',0),('--conversation-cache-mib',0),('--prompt-cache',0)]:flag(args,k,v)
 for k in ('--trim-stage-weights','--no-prefill-borrow','--no-kv-grow','--stop-eos'):
  if k not in args:args.append(k)
 flag(args,'--eos-ids','248044,248046');args[0]='/first-gdn/first-gdn-target'
 env={k:v for k,v in env.items()if not k.startswith('STRATA_ORIGINAL_MTP53')and k!='STRATA_ARTIFACT_IDENTITY_SHA256'}
 env.update(STRATA_LAYER0_Q8_DIAG='1',STRATA_LAYER0_Q8_DIAG_ARM='/results/num10.arm',STRATA_LAYER0_Q8_DIAG_DIR='/results/num10',STRATA_LAYER0_Q8_DIAG_BINDING_SHA256=sha(HERE/'prefill-first-gdn-observer-source-plan-v1.json'),STRATA_PREFILL_FIRST_GDN_DIR='/results/prefill_first_gdn',STRATA_PREFILL_FIRST_GDN_BINDING_SHA256=sha(HERE/'prefill-first-gdn-observer-source-plan-v1.json'),STRATA_ORIGINAL_MTP53_SERVING='0',STRATA_ONE_TOKEN_COMMIT='1',STRATA_QFUSE='0',STRATA_VERIFY_MIRROR_STAGE_T1='0',STRATA_COPY_ENGINE='0',STRATA_PF_GROUPED_F16='0',STRATA_PF_ESIMD_F16='0',STRATA_HOST_CRITICAL_PATH36='0',STRATA_CRITICAL_PATH_TRACE='0',STRATA_FIDELITY_DIAG='0',STRATA_PREFIX30='0',STRATA_PLE_INPUT33='0',STRATA_BATCH_FIDELITY_DIAG='0',STRATA_KV_GROW='0',ZE_AFFINITY_MASK='0,1')
 output=Path(output)
 if not output.is_absolute()or output.resolve()!=output or output.exists():raise ValueError('New canonical runtime intake required')
 record={'schema':1,'source_plan_sha256':sha(PLAN),'native_argv':args,'native_env':env,'cards':[0,1],'stages':[[0,0,24],[1,24,48]],'additional_readonly_mounts':{str(compile_root/'output'):'/first-gdn'},'actual_compile_receipt_sha256':receipt_sha,'actual_ELF_sha256':elf_sha,'actual58':b['actual58'],'scope':'Target-only first-row observed comparison, independent prefix1 NUM10 target; no original draft/cache/recurrent math reference substitution','mathematics_qualified':False,'speed_qualified':False,'GPU_executed':False,'model_payload_read':False};output.write_text(json.dumps(record,indent=2)+'\n',encoding='ascii');return record
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for k in ('compile-root','receipt-sha256','elf-sha256','base-recipe','base-recipe-sha256','output'):p.add_argument('--'+k,required=True)
 a=p.parse_args();print(json.dumps(prepare(Path(a.compile_root),a.receipt_sha256,a.elf_sha256,Path(a.base_recipe),a.base_recipe_sha256,Path(a.output)),ensure_ascii=True))
