"""Root-only fresh readonly H53 admission; no GPU execution or saved proof trust."""
import argparse,json
from pathlib import Path
from serial37_canonical_json_v3 import read_unique
from operation_pack_hash_witness_v2 import for_prepared
from sdk49_operation_scope_v1 import for_plan as sdk_for_plan
from batch53_logical_scope_v1 import for_plan as logical_for_plan
from audit_batch_numerical_suite_v53 import parent_arm

def finalized_binding(root):
 root=Path(root).resolve();plan=read_unique(root/'input-plan.snapshot.json');pack=for_prepared(Path(plan['prepared']),10800);sdk=sdk_for_plan(plan,10800);logical,roster,scope=logical_for_plan(plan);parent,actual,child=parent_arm(root,pack_epoch=pack,sdk_epoch=sdk,logical_epoch=logical,logical_roster=roster);logical.seal_predevice(current_roster=roster);pack.seal_predevice();sdk.seal_predevice();lp=logical.finalize(current_roster=roster);pp=pack.finalize();sp=sdk.finalize()
 if not lp['passed']:raise ValueError('Fresh readonly logical operation failed')
 return {'parent':parent,'actual_plan':actual,'actual_child':child,'current_logical_scope':scope,'current_logical_witness':lp,'current_pack_witness':pp,'current_sdk_witness':sp,'full_model_math_qualified':False,'serving_speed_qualified':False}
def main():
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();r=finalized_binding(a.root);print(json.dumps({'passed':r['parent']['passed'],'full_model_math_qualified':False}))
if __name__=='__main__':main()
