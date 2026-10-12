"""Compose original greedy-phase MAIN with a genuine fresh paired-expert object."""
import argparse,hashlib,json,subprocess
from pathlib import Path
import prepare_original_mtp53_greedy64_compile_v3 as original
HERE=Path(__file__).resolve().parent;PLAN=HERE/'original-mtp53-expert-pair-model-source-plan-v1.json'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for data in iter(lambda:f.read(8<<20),b''):h.update(data)
 return h.hexdigest()
def prepare(a):
 plan=json.loads(PLAN.read_bytes())
 for p,h in plan['files'].items():
  if sha(HERE.parents[1]/p)!=h:raise ValueError('Exact matched-pair model source changed '+p)
 for p,h in [(a.operator_receipt,a.operator_receipt_sha256),(a.operator_output/'iq_kernels.pair.o',a.iq_object_sha256),(a.operator_output/'libstrata_kernels.without_iq.a',a.filtered_archive_sha256)]:
  if sha(p)!=h:raise ValueError('Genuine fresh operator compilation changed '+str(p))
 receipt=json.loads(a.operator_receipt.read_bytes())
 if receipt.get('passed')is not True:raise ValueError('Actual successful fresh operator compile required')
 inventory=subprocess.run(['ar','t',str(a.operator_output/'libstrata_kernels.without_iq.a')],capture_output=True,check=True,text=True)
 if any(Path(x).name=='iq_kernels.dp.cpp.o'for x in inventory.stdout.splitlines()):raise ValueError('Original iq member must be absent from copied archive')
 record=original.prepare(a.sdk,a.sdk_receipt_sha256,a.output);cmd=record['compile_commands'][0]['argv']
 if cmd.count('/sdk/build/libstrata_kernels.a')!=1:raise ValueError('Exact original kernel archive seam required')
 cmd[cmd.index('/sdk/build/libstrata_kernels.a')]='/operator/libstrata_kernels.without_iq.a';cmd.insert(cmd.index('-Wl,--start-group'),'/operator/iq_kernels.pair.o')
 record['compile_commands'][0]['name']='original-mtp53-expert-pair-model';cmd[-1]='/out/original-mtp53-expert-pair-model'
 record['mounts']['operator_RO']=str(a.operator_output)
 record['pair_model_source_plan_sha256']=sha(PLAN);record['candidate_proposal_sha256']=plan['candidate_proposal_sha256'];record['fresh_operator_compile']={'receipt_path':str(a.operator_receipt),'receipt_sha256':sha(a.operator_receipt),'iq_path':str(a.operator_output/'iq_kernels.pair.o'),'iq_sha256':a.iq_object_sha256,'filtered_archive_path':str(a.operator_output/'libstrata_kernels.without_iq.a'),'filtered_archive_sha256':a.filtered_archive_sha256,'archive_iq_member_count':0,'root_compile_receipt':receipt}
 record['operator_GPU_PASS_required_before_model']=True;record['model_executed']=False;record['cache_policy_changed']=False
 original.component.write(a.output/'compile-recipe.json',record);return record
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--sdk',type=Path,required=True);p.add_argument('--sdk-receipt-sha256',required=True);p.add_argument('--operator-output',type=Path,required=True);p.add_argument('--operator-receipt',type=Path,required=True);p.add_argument('--operator-receipt-sha256',required=True);p.add_argument('--iq-object-sha256',required=True);p.add_argument('--filtered-archive-sha256',required=True);p.add_argument('--output',type=Path,required=True);print(json.dumps(prepare(p.parse_args()),ensure_ascii=True))
