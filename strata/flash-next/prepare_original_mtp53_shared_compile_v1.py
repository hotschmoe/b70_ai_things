"""Root-only compile-only recipe, no compiler/GPU/GGUF execution here."""
import argparse,hashlib,json
from pathlib import Path
import prepare_original_mtp53_two_stage_compile_v6 as parent
HERE=Path(__file__).resolve().parent
PLAN=HERE/'original-mtp53-shared-compile-source-plan-v1.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def compile_command(record):
 argv=list(record['compile_commands'][0]['argv']);source='/overlay/sycl/src/program/original_mtp53_two_stage_diagnostic.cpp'
 if argv.count(source)!=1 or argv.count('-Wl,--start-group')!=1:raise ValueError('Exact original V6 compile input required')
 base=argv[:argv.index('-Wl,--start-group')];base.remove(source)
 return [*base,'-I/sdk/source/sycl/src/kernels/cuda','-c','/overlay/sycl/src/kernels/original_mtp53_shared_sessions_compile.cpp','-o','/out/original-mtp53-shared-sessions.o']
def prepare(root,receipt,out):
 plan=json.loads(PLAN.read_bytes())
 for n,w in plan['files'].items():
  if sha(HERE.parents[1]/n)!=w:raise ValueError('Original shared compile source changed '+n)
 out=Path(out);record=parent.prepare(Path(root),receipt,out);overlay=out/'overlay'
 for source,target in plan['new_overlay_sources'].items():
  p=overlay/target;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((HERE.parents[1]/source).read_bytes())
 argv=compile_command(record)
 record.update(schema=2,shared_compile_source_plan_sha256=sha(PLAN),compile_commands=[{'name':'original-mtp53-shared-sessions.o','argv':argv}],overlay_sha256={str(p.relative_to(overlay)):sha(p)for p in sorted(overlay.rglob('*'))if p.is_file()},compile_only=True,linked=False,runtime_qualified=False,model_payload_read=False,scope='NoDri semantic C++ compile of shared original34/private4096 sessions; no runtime instantiation or model access')
 (out/'compile-recipe.json').write_text(json.dumps(record,indent=2)+'\n',encoding='ascii');return record
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--sdk',type=Path,required=True);p.add_argument('--sdk-receipt-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(prepare(a.sdk,a.sdk_receipt_sha256,a.output),ensure_ascii=True))
