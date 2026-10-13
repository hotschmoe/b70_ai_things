"""Root-only single-card isolated RMS oracle; independent inputs, current health/EOF."""
import argparse,json,os,re,subprocess,time
from pathlib import Path
from hc_rms3_owned_command_v1 import command,run_owned,require,sha,write
from build_hc_rms3_operator_v2 import binding,PLAN,ROOT
RUNTIME='sha256:c388186da30785b302c9c76c0ce8ca5e5351c9783c628177f6f7b9eed2f4ad17'
HEALTH='sha256:d55637b3353eaf470677627dc1627c3dda3ec6a6abb0298451aed34b73937067'
MARKER='HC_RMS3_EXPLICIT_OPERATOR_PASS original_inputs=1 baseline_observer_MAIN_match=1 explicit_own_normalized_RS_match=1 graphs=3 changed_input_bitwise=1 guards=1 whole48_qualified=0'
INPUTS={'norm.f32':40960,'own-residual_input.f32':40960,'own-attn_hc_normalized.f32':40960,'own-HC_square_sum.f32':16,'own-HC_rms_argument.f32':16,'own-HC_rms_scale.f32':16,'target-attn_hc_normalized.f32':40960}
def admission(a):
 source=binding();require(a.build.is_absolute()and a.build.resolve()==a.build and a.inputs.is_absolute()and a.inputs.resolve()==a.inputs,'Canonical actual build/inputs required');receipt=a.build/'receipt.json';require(sha(receipt)==a.build_receipt_sha256,'Actual fresh NoDri receipt required');b=json.loads(receipt.read_text());require(b['passed']is True and b['errors']==[]and b['devices_exposed']is False and b['GPU_operator_executed']is False and b['source_plan_sha256']==source,'Actual same-source owned NoDri success required');binary=a.build/'hc-rms3';require(binary.is_file()and not binary.is_symlink()and sha(binary)==a.elf_sha256 and b['outputs']=={'hc-rms3':a.elf_sha256},'Actual fresh compiler ELF required');pins={receipt:a.build_receipt_sha256,binary:a.elf_sha256,PLAN:source}
 for name,want in b['raw_sha256'].items():
  p=a.build/name;require(p.is_file()and not p.is_symlink()and sha(p)==want,'Actual compile raw closure differs');pins[p]=want
 p=a.inputs/'report.json';require(sha(p)==a.inputs_report_sha256,'Exact root-owned original input export receipt required');r=json.loads(p.read_text());require(r['passed']is True and r['errors']==[]and r['original_norm_only']is True and r['target_arrays_are_comparison_only']is True and r['native_values_used_as_primary_inputs']is False and r['postCPU_full4']['passed']is True,'Independent original norm/input closing receipt required');pins[p]=a.inputs_report_sha256
 require(set(r['artifacts'])=={p.name for p in a.inputs.iterdir()if p.is_file()and p.name!='report.json'},'Original input complete saved raw roster required')
 for n,want in r['artifacts'].items():
  p=a.inputs/n;require(p.is_file()and not p.is_symlink()and sha(p)==want,'Original input byte closure differs '+n);pins[p]=want
 for n,size in INPUTS.items():require(n in r['artifacts']and(a.inputs/n).stat().st_size==size,'Bounded original input shape differs '+n)
 require(re.fullmatch(r'[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]',a.expected_pci)and Path('/sys/class/drm/card0/device').resolve().name==a.expected_pci,'Actual leased card0 PCI metadata differs');return pins

def native_command(a,out,name):
 mounts=[str(a.build)+':/harness:ro',str(a.inputs)+':/inputs:ro',str(out)+':/out:rw'];argv,expected=command(RUNTIME,name,'b70.hc.rms3.runtime',mounts,GPU=True)
 for env in('ZE_AFFINITY_MASK=0','ONEAPI_DEVICE_SELECTOR=level_zero:gpu','SYCL_CACHE_PERSISTENT=0','SYCL_UR_TRACE=0','B70_HC_RMS3_OBSERVE=1','B70_HC_RMS3_EXPLICIT_SQRT_DIVIDE=1','B70_HC_EXPECTED_PCI='+a.expected_pci):argv+=['-e',env]
 leaf='set -Eeo pipefail; test -e /dev/dri; if [[ ${SETVARS_COMPLETED:-0} != 1 ]]; then source /opt/intel/oneapi/setvars.sh > /out/setvars.log 2>&1; fi; ldd /harness/hc-rms3 > /out/runtime-ldd.txt; if grep -q "not found" /out/runtime-ldd.txt; then exit 90; fi; exec /harness/hc-rms3 /inputs /out/oracle'
 argv+=['--entrypoint','/bin/bash',RUNTIME,'-lc',leaf];expected['env']={argv[i+1].split('=',1)[0]:argv[i+1].split('=',1)[1]for i,v in enumerate(argv[:-1])if v=='-e'};return argv,expected

def main(a):
 pins=admission(a);require(os.path.samefile('/proc/self/fd/8','/mnt/vm_8tb/b70/gpu.lock.0'),'Inherited root card0 FD8 required');out=a.output;require(out.is_absolute()and out.resolve()==out and not out.exists(),'New canonical runtime output required');out.mkdir();report={'schema':1,'passed':False,'errors':[],'started_epoch':time.time(),'source_plan_sha256':sha(PLAN),'build_receipt_sha256':a.build_receipt_sha256,'elf_sha256':a.elf_sha256,'input_report_sha256':a.inputs_report_sha256,'card':0,'expected_PCI':a.expected_pci,'actual_MAIN_RMS_observed':False,'whole48_qualification':False,'model_payload_or_provider_read':False,'isolated_original_operands':True};closed=False;attempted=False;parent_signals=[]
 import signal
 saved_signals={n:signal.signal(n,lambda number,frame:parent_signals.append(number))for n in(signal.SIGINT,signal.SIGTERM,signal.SIGHUP)}
 def checked(argv,label,timeout=180):
  write(out/(label+'.command.json'),argv);started=time.time()
  with(out/(label+'.stdout')).open('wb')as so,(out/(label+'.stderr')).open('wb')as se:r=subprocess.run(argv,stdout=so,stderr=se,timeout=timeout,pass_fds=(8,))
  write(out/(label+'.result.json'),{'return_code':r.returncode,'started_epoch':started,'finished_epoch':time.time(),'stdout_sha256':sha(out/(label+'.stdout')),'stderr_sha256':sha(out/(label+'.stderr'))});require(r.returncode==0,label+' failed')
 try:
  checked(['journalctl','-k','--since','@'+str(int(report['started_epoch'])),'--no-pager'],'pre-kernel');checked([str(ROOT/'bin/xpu-health'),'--card','0','--img',HEALTH],'pre-health');require(not parent_signals,'Signal before native launch');argv,expected=native_command(a,out,'b70-hc-rms3-runtime-'+str(os.getpid()));attempted=True;owned=run_owned(argv,out,expected,timeout=180);report['owned']=owned;closed=owned['owned_removed']and owned['original_EOF'];report['errors'].extend(owned['errors']);require(owned['passed'],'Actual RMS operator failed');require((out/'stdout').read_text(encoding='ascii').count(MARKER)==1,'Complete exact RMS3/graph/guard terminal marker required')
  for name,size in{'baseline-normalized.f32':40960,'observed-normalized.f32':40960,'native-square_sum.f32':16,'native-rms_argument.f32':16,'native-rms_scale.f32':16,'explicit-sqrt-divide-normalized.f32':40960,'explicit-sqrt-divide-rms-scale.f32':16}.items():
   p=out/'oracle'/name;require(p.is_file()and not p.is_symlink()and p.stat().st_size==size,'Complete finite RMS operator output required '+name)
   import numpy as np
   require(np.isfinite(np.frombuffer(p.read_bytes(),dtype='<f4')).all(),'Nonfinite isolated oracle output')
  require((out/'oracle/baseline-normalized.f32').read_bytes()==(out/'oracle/observed-normalized.f32').read_bytes()==(a.inputs/'target-attn_hc_normalized.f32').read_bytes(),'Original before/after and saved actualMAIN normalized byte match required');require((out/'oracle/explicit-sqrt-divide-normalized.f32').read_bytes()==(a.inputs/'own-attn_hc_normalized.f32').read_bytes()and(out/'oracle/explicit-sqrt-divide-rms-scale.f32').read_bytes()==(a.inputs/'own-HC_rms_scale.f32').read_bytes(),'Explicit operator must equal own independent normalized/RS');report['RMS3_isolated_outputs_passed']=True;report['explicit_sqrt_divide_own_output_match']=True
 except BaseException as exc:report['errors'].append(type(exc).__name__+': '+str(exc))
 finally:
  # run_owned never returns while original creator/container/EOF is unresolved.
  # Pre-launch failure has no actor; after-launch health requires actual absence.
  if not attempted or closed:
   try:checked([str(ROOT/'bin/xpu-health'),'--card','0','--img',HEALTH],'post-health');report['post_health_passed']=True
   except BaseException as exc:report['errors'].append('post-health: '+str(exc))
   try:
    checked(['journalctl','-k','--since','@'+str(int(report['started_epoch'])),'--no-pager'],'post-kernel');require(not re.search(r'Fault response|Faulted Address|Engine reset|Timedout job|segfault|GPU HANG|CAT error',(out/'post-kernel.stdout').read_text(),re.I),'Fresh kernel fault');report['kernel_window_passed']=True
   except BaseException as exc:report['errors'].append('kernel: '+str(exc))
  else:report['errors'].append('Original actor absence/EOF not proven; health forbidden')
  for p,want in pins.items():
   try:require(sha(p)==want,'Closing original byte/source changed '+str(p))
   except BaseException as exc:report['errors'].append('input-close: '+str(exc))
  try:require(binding()==report['source_plan_sha256'],'Closing RMS lifecycle source changed')
  except BaseException as exc:report['errors'].append('source-close: '+str(exc))
  if parent_signals:report['errors'].append('Signals latched '+repr(parent_signals))
  for number,handler in saved_signals.items():signal.signal(number,handler)
  report.update(passed=not report['errors'],finished_epoch=time.time(),artifacts={str(p.relative_to(out)):sha(p)for p in out.rglob('*')if p.is_file()and p.name!='report.json'});write(out/'report.json',report)
 print(json.dumps({'passed':report['passed'],'errors':report['errors'],'report':str(out/'report.json')},ensure_ascii=True));return 0 if report['passed']else 1
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for n in('build','inputs','output'):p.add_argument('--'+n,type=Path,required=True)
 for n in('build-receipt-sha256','elf-sha256','inputs-report-sha256','expected-pci'):p.add_argument('--'+n,required=True)
 raise SystemExit(main(p.parse_args()))
