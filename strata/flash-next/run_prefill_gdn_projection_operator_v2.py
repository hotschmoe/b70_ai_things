"""Runtime-only V2: guarded setvars startup, actual immutable V1 ELF/export."""
import argparse,json,os,re,shlex,subprocess,time
from pathlib import Path
import prepare_prefill_gdn_projection_operator_v1 as source
from hc_rms3_owned_command_v1 import command,run_owned,require,sha as original_sha,write
def sha(p):
 p=Path(p);require(p.is_file()and not p.is_symlink(),"Regular original artifact required "+str(p));return original_sha(p)
PLAN=source.HERE/'prefill-gdn-projection-runtime-source-plan-v2.json'
BUILD_PLAN=source.HERE/'prefill-gdn-projection-operator-source-plan-v1.json'
RUNTIME='sha256:c388186da30785b302c9c76c0ce8ca5e5351c9783c628177f6f7b9eed2f4ad17'
HEALTH='sha256:d55637b3353eaf470677627dc1627c3dda3ec6a6abb0298451aed34b73937067'
MARKER='PF_GDN_OPERATOR_PASS original_weights=1 synthetic_F32_operands=1 direct_cases=4 graphs=2 changed_input=1 bitwise_fields=5 last_row_packets=2 guards=1 conv_recurrence_qualified=0 whole_model_qualified=0'
SIZES={'qkv.q80':27852800,'z.q80':16711680,'out.q80':16711680,'alpha.bf16':245760,'beta.bf16':245760,'dt.f32':192,'ssm_a.f32':192}
def binding():
 before=sha(PLAN)
 for n,h in json.loads(PLAN.read_text())['files'].items():require(sha(source.ROOT/n)==h,'Projection fixture source changed '+n)
 require(sha(PLAN)==before,'Projection fixture plan changed');return before

def prepare_admit(a):
 rpath=a.prepared/'compile-recipe.json';require(sha(rpath)==a.recipe_sha256,'Exact source-prepared recipe required');r=json.loads(rpath.read_text());pins={rpath:a.recipe_sha256,PLAN:binding()};require(r['SDK']==str(source.SDK)and r['SDK_receipt_sha256']==source.RECEIPT_SHA and r['compiled_or_GPU_executed']is False,'Actual62 source-only intake required');require(set(r['files'])=={'fixture.cpp','prefill_gdn_decoder_projection_v1.hpp','prefill_gdn_post_store_generated_v1.inc'},'Exact source roster required')
 for name,h in r['files'].items():p=a.prepared/name;require(sha(p)==h,'Prepared source changed');pins[p]=h
 for name,h in r['source_pins'].items():p=Path(name);require(sha(p)==h,'Original source/compiler metadata changed');pins[p]=h
 require((a.prepared/'fixture.cpp').read_bytes()==(source.ROOT/'kernels/prefill_gdn_decoder_projection_fixture_v1.cpp').read_bytes(),'Frozen fixture differs');require((a.prepared/'prefill_gdn_decoder_projection_v1.hpp').read_bytes()==(source.HERE/'original-mtp53/prefill_gdn_decoder_projection_v1.hpp').read_bytes(),'Candidate helper differs');require((a.prepared/'prefill_gdn_post_store_generated_v1.inc').read_text()==source.extract((source.SDK/'source/sycl/src/prefill/kernels.dp.cpp').read_text()),'Exact post-store source extraction differs');require((r['compile_argv'],r['link_argv'])==source.commands((source.SDK/'build/incremental-parent-commands.txt').read_text()),'Actual production argv differs')
 for name,h in r['archives_sha256'].items():p=source.SDK/'build'/name;require(sha(p)==h,'Actual SDK archive bytes differ');pins[p]=h
 return r,pins

def run(a):
 require(a.mode=="runtime","V2 is runtime-only; immutable V1 build/export reused")
 require(__debug__ and not os.environ.get('PYTHONOPTIMIZE'),'Unoptimized owned controller required');require(os.path.samefile('/proc/self/fd/8','/mnt/vm_8tb/b70/gpu.lock.0'),'Root inherited card0 lease required');source_sha=binding();out=a.output.resolve();require(not out.exists(),'New output required');out.mkdir();r={'schema':1,'passed':False,'errors':[],'mode':a.mode,'started_epoch':time.time(),'source_plan_sha256':source_sha,'GPU_operator_executed':a.mode=='runtime','devices_exposed':a.mode=='runtime','model_payload_or_provider_read':False,'conv_recurrence_qualified':False,'whole_model_qualified':False};pins={};attempted=False;closed=False
 def checked(cmd,label):
  write(out/(label+'.command.json'),cmd);start=time.time()
  with(out/(label+'.stdout')).open('wb')as so,(out/(label+'.stderr')).open('wb')as se:p=subprocess.run(cmd,stdout=so,stderr=se,timeout=180,pass_fds=(8,))
  write(out/(label+'.result.json'),{'return_code':p.returncode,'started_epoch':start,'finished_epoch':time.time()});require(p.returncode==0,label+' failed')
 try:
  if a.mode=='build':
   recipe,pins=prepare_admit(a);write(out/'recipe.snapshot.json',recipe);r['recipe_sha256']=a.recipe_sha256;image=recipe['image'];leaf='\n'.join(['set -Eeo pipefail','test ! -e /dev/dri','if [[ ${SETVARS_COMPLETED:-0} != 1 ]]; then source /opt/intel/oneapi/setvars.sh > /out/setvars.log 2>&1; fi','icpx --version > /out/compiler-version.txt','dpkg-query -W > /out/packages.txt','cmp /sdk/build/compiler-version.txt /out/compiler-version.txt','cmp /sdk/build/packages.txt /out/packages.txt',shlex.join(recipe['compile_argv']),shlex.join(recipe['link_argv']),'ldd /out/pf-gdn > /out/ldd.txt','! grep -q "not found" /out/ldd.txt'])+'\n';(out/'compile.sh').write_text(leaf,encoding='ascii');mounts=[str(a.prepared.resolve())+':/inputs:ro',str(source.SDK)+':/sdk:ro',str(source.SDK/'source')+':/src:ro',str(out)+':/out:rw'];argv,expected=command(image,'b70-pf-gdn-build-'+str(os.getpid()),'b70.pf.gdn.build',mounts);argv+=['--entrypoint','/bin/bash',image,'-lc','exec bash /out/compile.sh']
  else:
   require(a.build and a.inputs and a.build_receipt_sha256 and a.elf_sha256 and a.inputs_report_sha256,'Runtime exact build/export arguments required');bp=a.build/'report.json';require(sha(bp)==a.build_receipt_sha256,'Exact actual NoDri build report required');br=json.loads(bp.read_text());require(br['passed']is True and br['errors']==[]and br['mode']=='build'and br['source_plan_sha256']==sha(BUILD_PLAN),'Actual same-source build required');require(br['outputs']=={'fixture.o':sha(a.build/'fixture.o'),'pf-gdn':a.elf_sha256}and sha(a.build/'pf-gdn')==a.elf_sha256,'Fresh object/ELF mismatch');pins={bp:a.build_receipt_sha256,a.build/'pf-gdn':a.elf_sha256}
   for name,h in br['artifacts'].items():p=a.build/name;require(sha(p)==h,'Actual build raw closure differs');pins[p]=h
   ip=a.inputs/'report.json';require(sha(ip)==a.inputs_report_sha256,'Exact original export report required');ir=json.loads(ip.read_text());require(ir['passed']is True and ir['errors']==[]and ir['original_weight_export_only']is True and ir['native_values_used_as_primary_inputs']is False and ir['postCPU_full4']['passed']is True,'Original provider export closing evidence required');require(set(ir['roles'])==set(SIZES),'Complete original export roles required');require(set(ir['artifacts'])=={str(p.relative_to(a.inputs))for p in a.inputs.rglob('*')if p.is_file()and p.name!='report.json'},'Complete export raw roster required');pins[ip]=a.inputs_report_sha256
   for name,h in ir['artifacts'].items():p=a.inputs/name;require(sha(p)==h,'Original export bytes changed');pins[p]=h
   for name,size in SIZES.items():require((a.inputs/name).stat().st_size==size and ir['roles'][name]['sha256']==sha(a.inputs/name),'Original export size/digest differs')
   pci=Path('/sys/class/drm/card0/device').resolve().name;require(pci==a.expected_pci,'Actual card0 PCI metadata differs');checked([str(source.ROOT/'bin/xpu-health'),'--card','0','--img',HEALTH],'pre-health');image=RUNTIME;mounts=[str(a.build.resolve())+':/harness:ro',str(a.inputs.resolve())+':/inputs:ro',str(out)+':/out:rw'];argv,expected=command(image,'b70-pf-gdn-runtime-'+str(os.getpid()),'b70.pf.gdn.runtime',mounts,GPU=True);env={'ZE_AFFINITY_MASK':'0','ONEAPI_DEVICE_SELECTOR':'level_zero:gpu','SYCL_CACHE_PERSISTENT':'0','SYCL_UR_TRACE':'0','STRATA_PF_GDN_DECODER_PROJ':'1','B70_PF_GDN_EXPECTED_PCI':pci};expected['env']=env
   for k,v in env.items():argv+=['-e',k+'='+v]
   argv+=['--entrypoint','/bin/bash',image,'-lc','set -Eeo pipefail; if [[ ${SETVARS_COMPLETED:-0} != 1 ]]; then source /opt/intel/oneapi/setvars.sh > /out/setvars.log 2>&1; fi; ldd /harness/pf-gdn > /out/ldd.txt; ! grep -q "not found" /out/ldd.txt; exec /harness/pf-gdn /inputs']
  attempted=True;owned=run_owned(argv,out,expected,timeout=300);r['owned']=owned;closed=owned['owned_removed']and owned['original_EOF'];r['errors'].extend(owned['errors']);require(owned['passed'],'Owned projection operator build/runtime failed')
  if a.mode=='build':r['outputs']={n:sha(out/n)for n in('fixture.o','pf-gdn')}
  else:require((out/'stdout').read_text(encoding='ascii').count(MARKER)==1,'Exact bitwise/graph/guard terminal proof required')
 except BaseException as e:r['errors'].append(type(e).__name__+': '+str(e))
 finally:
  if a.mode=='runtime'and(not attempted or closed):
   try:checked([str(source.ROOT/'bin/xpu-health'),'--card','0','--img',HEALTH],'post-health');checked(['journalctl','-k','--since','@'+str(int(r['started_epoch'])),'--no-pager'],'post-kernel');require(not re.search(r'Fault response|Faulted Address|Engine reset|Timedout job|segfault|GPU HANG|CAT error',(out/'post-kernel.stdout').read_text(),re.I),'Fresh kernel fault')
   except BaseException as e:r['errors'].append('health/kernel: '+str(e))
  for p,h in pins.items():
   try:require(sha(p)==h,'Consumed input/source bytes changed '+str(p))
   except BaseException as e:r['errors'].append('input-close: '+str(e))
  try:require(binding()==source_sha,'Closing source changed')
  except BaseException as e:r['errors'].append('source-close: '+str(e))
  r.update(passed=not r['errors'],finished_epoch=time.time(),artifacts={str(p.relative_to(out)):sha(p)for p in out.rglob('*')if p.is_file()and p.name!='report.json'});write(out/'report.json',r)
 print(json.dumps({'passed':r['passed'],'errors':r['errors'],'report':str(out/'report.json')},ensure_ascii=True));return 0 if r['passed']else 1
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['runtime']);p.add_argument('--output',type=Path,required=True)
 for n in('prepared','build','inputs'):p.add_argument('--'+n,type=Path)
 for n in('recipe-sha256','build-receipt-sha256','elf-sha256','inputs-report-sha256','expected-pci'):p.add_argument('--'+n)
 raise SystemExit(run(p.parse_args()))
