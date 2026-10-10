#!/usr/bin/env python3
"""Readonly closed-evidence V8 adjudication: explicitly derived missing childSHA.
Original successful parent/child/source/captures are never rewritten.
"""
import argparse,copy,hashlib,json,os,sys
from pathlib import Path
import validate_batch_serial_cacheoff_v8 as frozen_reader
import batch_serial_cacheoff_v8 as ctrl
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
PLAN=HERE/'cacheoff-v8-adjudication-and-v9-source-plan.json'
sha=ctrl.sha;read=ctrl.read;require=ctrl.require

def tree_binding(root):
 root=Path(root).resolve();files={}
 for path in sorted(root.rglob('*')):
  require(not path.is_symlink(),'Closed original evidence symlink refused')
  if not path.is_file():continue
  before=path.stat();digest=sha(path);after=path.stat();sig=lambda x:[x.st_dev,x.st_ino,x.st_size,x.st_mtime_ns,x.st_ctime_ns];require(sig(before)==sig(after),'Closed original artifact changed during scan');files[str(path.relative_to(root))]={'sha256':digest,'bytes':after.st_size,'stat5':sig(after)}
 require(files,'Closed original evidence tree empty');return files

def source_binding():
 plan=read(PLAN)
 for name,want in plan['files'].items():require(sha(ROOT/name)==want,'Frozen closed-adjudication/source supplement changed '+name)
 return plan['files']

def derive_child_view(root,parent,child):
 root=Path(root).resolve();require('plan_sha256' not in child,'Only exact missing producer child_plan_SHA case can be adjudicated');inputpath=root/'input-plan.snapshot.json';snapshot=root/'child/plan.snapshot.json';external=Path(parent['plan']).resolve();original=read(inputpath);require(original==read(snapshot)==read(external),'Actual external/input/child loaded plan content differs');digest=sha(snapshot);require(digest==sha(inputpath)==sha(external)==parent['plan_sha256'],'Actual external/input/child snapshot SHA association differs');require(parent['child_report_sha256']==sha(root/'child/report.json'),'Original parent-to-child report hash differs')
 expected=['/usr/bin/python3',str(Path(ctrl.__file__).resolve()),'run','--plan',str(external),'--pre-health',str(root/'pre-health.json'),'--output',str(root/'child')];actual=read(root/'child.command.json');require(actual==expected,'Actual child CLI did not load the recorded original plan')
 require(sha(root/'controller.py')==sha(Path(ctrl.__file__))==original['driver_sha256']==parent['controller_sha256'],'Actual original controller snapshot/source differs');wrapper=HERE/'qualify_batch_serial_cacheoff_v8.py';require(sha(root/'wrapper.py')==sha(wrapper)==parent['wrapper_sha256'],'Actual original wrapper snapshot/source differs')
 view=copy.deepcopy(child);view['plan_sha256']=digest;return view,{'original_child_field_absent':True,'derived_field':'plan_sha256','derived_value':digest,'derived_from_real_snapshot':str(snapshot),'snapshot_SHA256':digest,'original_report_sha256':sha(root/'child/report.json'),'producer_emitted_missing_field':False,'actual_child_command':actual,'original_producer_interpreter_path':actual[0],'current_adjudicator_interpreter_is_not_original_runtime_authority':True,'original_interpreter_ELF_hash_recorded_by_V8':False,'metadata_measurements_or_raw_flags_changed':False}

def adjudicate(root):
 root=Path(root).resolve();dependencies=source_binding();before=tree_binding(root);parent=read(root/'parent-qualification.json');child=read(root/'child/report.json');require(parent['passed'] is True and child['passed'] is True,'Original successful V8 parent/child required, never repair failed runs');view,derivation=derive_child_view(root,parent,child);original_read=ctrl.read
 def closed_read(path):
  if Path(path).resolve()==root/'child/report.json':
   require(sha(path)==derivation['original_report_sha256'] and original_read(path)==child,'Original successful child changed during derived view');return copy.deepcopy(view)
  return original_read(path)
 try:
  ctrl.read=closed_read
  checked_parent,checked_view,plan,binding=frozen_reader.finalized_binding(root)
 finally:ctrl.read=original_read
 require(checked_parent==parent and checked_view==view,'Strict reader returned a different original/derived view');after=tree_binding(root);require(before==after,'Original evidence tree changed during readonly adjudication');source_binding()
 return {'schema':1,'adjudicated_passed':True,'scope':'closed original V8 actual numerical source196, explicitly derived missing childplanSHA; no original receipt mutation','original_parent_passed':parent['passed'],'original_child_passed':child['passed'],'original_parent_sha256':sha(root/'parent-qualification.json'),'original_child_sha256':sha(root/'child/report.json'),'derived_evidence_view':derivation,'strict_actual_binding':binding,'actual_matched_vector_pairs':binding['matched_vector_pairs'],'cache_qualification_granted':False,'full_model_math_qualified':False,'source_dependencies':dependencies,'original_tree_before':before,'original_tree_after':after,'original_tree_unchanged':True,'GPU_or_Docker_or_model_inference_executed':False}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.run_root.resolve();out=a.output.resolve();require(not out.exists() and out!=root and root not in out.parents,'NEW adjudication must live outside immutable original evidence');r=adjudicate(root);out.write_text(json.dumps(r,indent=2)+'\n',encoding='ascii');print(json.dumps({'adjudicated_passed':r['adjudicated_passed'],'actual_pairs':r['actual_matched_vector_pairs'],'original_tree_unchanged':True}));return 0
if __name__=='__main__':raise SystemExit(main())
