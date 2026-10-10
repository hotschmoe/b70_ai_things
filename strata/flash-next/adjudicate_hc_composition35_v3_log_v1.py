"""NEW read-only view for one exact source35 UR-interleaved FP capability log.
All frozen V3 source/runtime/raw/current guards run. No evidence is rewritten.
"""
import argparse,hashlib,json,re
from pathlib import Path
import validate_hc_composition35_v3 as frozen
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
SOURCE_PLAN=HERE/'hc-composition35-v3-log-adjudication-source-plan-v1.json'
RUN_ROOT=Path('/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f16-source35-20261009/hc-composition35-v3-run')
PINS={'parent-qualification.json':'92b165a637d2c329cae740c0442b14abcdb7148db591aa7735d48ab3fab970c1','raw-component-proof.json':'fbcded1144f210b48bbdeb62f00e24b698fb0ab2ebe0913b7e3b8576841a5e5c','usm-logical-free-proof.json':'38a60e48c5965ae383c8aee636b9680bce7c68771f54eab7d25dab0e62264732','leaf.log':'67511750719197190310c49b2251c890484ed5b6029d5480a9e532e187e0ac28'}
sha,read,require=frozen.sha,frozen.read,frozen.require
PREFIX='HC35_COMPOSITION_FP_CONFIG flags=   ---> urDeviceGetInfo'
SCOPE=' observed_device_flags_only=1 compiler_lowering_unobserved=1'
CAPABILITIES=('CORRECTLY_ROUNDED_DIVIDE_SQRT','ROUND_TO_NEAREST','ROUND_TO_ZERO','ROUND_TO_INF','INF_NAN','DENORM','FMA')
QUERY=re.compile(r'   <--- urDeviceGetInfo\(\.hDevice = (0x[0-9a-fA-F]+), \.propName = UR_DEVICE_INFO_SINGLE_FP_CONFIG, \.propSize = 4, \.pPropValue = (0x[0-9a-fA-F]+) \(([^()]*)\), \.pPropSizeRet = nullptr\) -> UR_RESULT_SUCCESS;')

def source_binding():
 plan=read(SOURCE_PLAN)
 for name,want in plan['files'].items():require(sha(ROOT/name)==want,'Frozen adjudication source changed '+name)
 return plan['files']
def known_interleaved_observation(text):
 lines=text.splitlines();positions=[i for i,l in enumerate(lines) if l.startswith('HC35_COMPOSITION_FP_CONFIG ')];require(len(positions)==1,'Exactly one original FP prefix required');i=positions[0]
 require(0<i and i+3<len(lines) and lines[i]==PREFIX and lines[i-1].startswith('HC35_COMPOSITION_DEVICE '),'Only exact bounded known prefix/device ordering admitted');match=QUERY.fullmatch(lines[i+1]);require(match is not None,'Only one complete successful SINGLE_FP_CONFIG UR insertion admitted')
 names=tuple(x.removeprefix('UR_DEVICE_FP_CAPABILITY_FLAG_') for x in match.group(3).split(' | '));require(names==CAPABILITIES,'Known source capability names/order differ')
 require(lines[i+2]=='32,16,2,4,8,64,65,'+SCOPE and lines[i+3].startswith('HC35_COMPOSITION_CONFIG '),'Exact completed source enum list/scope/config ordering required')
 prior=[line for line in lines[:i] if line.startswith('   <--- urDeviceGetInfo(') and '.propName = UR_DEVICE_INFO_DRIVER_VERSION,' in line];require(prior and '.hDevice = '+match.group(1)+',' in prior[-1] and prior[-1].endswith(' -> UR_RESULT_SUCCESS;'),'FP query device differs from original current driver query')
 return {'original_line_indices_0based':[i,i+1,i+2],'original_lines':lines[i:i+3],'observed_device_handle':match.group(1),'observed_enum_values':[32,16,2,4,8,64,65],'observed_capability_names':list(names),'joined_FP_CONFIG':'HC35_COMPOSITION_FP_CONFIG flags='+lines[i+2],'normalization_scope':'NEW view only: exact3-line FP prefix/successful UR query/enum suffix; originallog untouched','device_general_FP_mode_qualified':False}
def joined_leaf_binding(root,saved):
 root=Path(root).resolve();require(saved['log_sha256']==sha(root/'leaf.log'),'Original leaf log SHA changed');text=(root/'leaf.log').read_text();view=known_interleaved_observation(text);lines=text.splitlines();result={}
 for kind in ('DEVICE','CONFIG'):
  matching=[l for l in lines if l.startswith('HC35_COMPOSITION_'+kind+' ')];require(len(matching)==1,'Actual unique original device/config line missing');result[kind]=matching[0]
 result['FP_CONFIG']=view['joined_FP_CONFIG'];require(result['CONFIG']=='HC35_COMPOSITION_CONFIG synthetic=1 compiled_composition=1 shadows_separate=1 subgroup=32 eps=1e-6 graph_and_queue=1 normal_model_graph_qualified=0 device_intrinsics_qualified=0 model_math_qualified=0','Original configuration changed');require('backend=level_zero ' in result['DEVICE'] and ' affinity=0 selector=level_zero:gpu' in result['DEVICE'],'Original device/pin changed')
 return {'log_sha256':sha(root/'leaf.log'),'lines':result,'original_UR_insertion_observation':view,'device_general_FP_mode_qualified':False}
def tree_binding(root):
 root=Path(root).resolve();rows={}
 for path in sorted(root.rglob('*')):
  require(not path.is_symlink(),'Preserved source/evidence tree symlink refused')
  if path.is_file():
   before=path.stat();digest=sha(path);after=path.stat();sig=lambda s:[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns];require(sig(before)==sig(after),'Original file changed while binding');rows[str(path.relative_to(root))]={'sha256':digest,'stat5':sig(after)}
 return rows
def finalized_binding(run_root):
 root=Path(run_root).resolve();require(root==RUN_ROOT.resolve(),'Only exact completed known V3 root admitted');source=source_binding()
 for name,want in PINS.items():require(sha(root/name)==want,'Original exact V3 evidence changed '+name)
 before=tree_binding(root);original_error=None
 try:frozen.finalized_binding(root)
 except ValueError as error:original_error=str(error)
 require(original_error=='FP capability observation scope changed','Only original known readonly format failure accepted; other gates cannotbe waived')
 original=frozen.leaf_log_binding
 try:
  frozen.leaf_log_binding=joined_leaf_binding
  parent,raw,binding=frozen.finalized_binding(root)
 finally:frozen.leaf_log_binding=original
 require(tree_binding(root)==before and source_binding()==source,'Original tree/source changed during readonly adjudication')
 for name,want in PINS.items():require(sha(root/name)==want,'Original evidence changed during readonly admission')
 return parent,raw,{'adjudication_kind':'NEW_exact_UR_interleaved_FP_log_view','original_reader_failed':True,'original_reader_error':original_error,'original_reader_success_claimed':False,'original_parent_passed':parent['passed'],'original_evidence_rewritten':False,'original_tree_before_after_unchanged':True,'original_pins':PINS,'V3_all_other_finalized_gates':binding,'new_source_binding':source,'device_intrinsics_qualified':False,'normal_model_graph_qualified':False,'full_model_math_qualified':False,'tolerance_gate':None}
def main():
 p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();root=a.run_root.resolve();require(not out.exists() and not out.is_relative_to(root),'NEW output outside preserved V3 tree required');parent,raw,binding=finalized_binding(root);out.mkdir(parents=True);(out/'adjudication.json').write_text(json.dumps(binding,indent=2,ensure_ascii=True)+'\n',encoding='ascii');print(json.dumps({'readonly_adjudication_passed':True,'original_reader_success_claimed':False,'normal_model_graph_qualified':False}));return 0
if __name__=='__main__':raise SystemExit(main())
