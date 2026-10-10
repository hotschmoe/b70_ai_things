"""Source/current closed V8 prerequisite and own arithmetic work recollection."""
import hashlib,json,re,struct
from pathlib import Path
import numpy as np
import qualify_native_rms_rsqrt37_v8 as prior
from serial37_canonical_json_v3 import canonical,read_unique
from owned_hc_device_rs_protocol_v1 import require,response
HERE=Path(__file__).parent;ROOT=HERE.parents[1];PLAN=HERE/'owned-hc-device-rs-source-plan-v2.json'
PRIOR_ROOT=Path('/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/native-rms-rsqrt37-owned-v8-run-v1')
EXTERNAL=PRIOR_ROOT.parent/'native-rms-rsqrt37-owned-v8-readonly-binding-v1.json'
REPORT_SHA='8f070f780a63eae2d92abdd98e62d2ecfc36ab1a95a81ade055d3b184fdf9c90'
EXTERNAL_SHA='3129cfd8c78a29e3cf9bd6f8792d9f93154439efe92c950ab871afbd5b9902bc'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def source_binding():
 row=read_unique(PLAN)
 for n,w in row['files'].items():require(sha(ROOT/n)==w,'Own reference source changed '+n)
 require(sha(prior.PLAN)=='4ed12398b46e490ea1a85fef902f1441e76f42d83dcfc818c179b82ee735108c','Original owned V8 source changed');prior.source_binding();return {'source_plan_sha256':sha(PLAN),'operation':'sycl::rsqrt','original_NON_HC_changed':False,'captured_operands_or_lookup_used':False}
def prior_header(root):require(Path(root).resolve()==PRIOR_ROOT and sha(PRIOR_ROOT/'report.json')==REPORT_SHA and sha(EXTERNAL)==EXTERNAL_SHA,'Exact closed V8 evidence prerequisite required')
def prior_binding(root):
 prior_header(root);binding=prior.finalized_binding(root);require(binding['report_sha256']==REPORT_SHA and canonical(binding)==canonical(read_unique(EXTERNAL)),'Actual current V8 contradicts original independent admission');return {'root':str(PRIOR_ROOT),'binding':binding,'external_sha256':EXTERNAL_SHA,'captured_values_output_targets_only':True}
def read_bound_array(binding,shape,work_root):
 path=Path(binding['path']);before=path.stat();raw=path.read_bytes();after=path.stat();require(path.resolve().parent==Path(work_root).resolve() and not path.is_symlink() and before==after and hashlib.sha256(raw).hexdigest()==binding['sha256'] and path.read_bytes()==raw and path.stat()==after and len(raw)==binding['bytes']==4*np.prod(shape),'Original exact consumed own array changed/escaped');a=np.frombuffer(raw,dtype='<f4').reshape(shape);require(np.isfinite(a).all(),'Own array nonfinite');return a
def recollect(log,phase,work,work_root=None):
 semantic=[line for line in Path(log).read_text().splitlines(keepends=True) if line.startswith('OWNRS37_')];records=phase['records'];require(len(records) in (1,386),'Exact first-only or first+prefix4 request roster required');require(len(semantic)==len(records)+2 and semantic[0]==phase['ready'] and semantic[-1]==phase['done'],'Original own helper marker order/coverage differs');require(re.fullmatch(r'OWNRS37_READY pid=[1-9][0-9]* backend=level_zero affinity=0 width=4 maximum=512\n',semantic[0]) is not None and semantic[-1]==f'OWNRS37_DONE requests={len(records)} graph_retired=1 owned_allocations_freed=1\n','Actual own helper READY/free/graph terminal differs')
 for seq,(row,line) in enumerate(zip(records,semantic[1:-1]),1):
  a=np.frombuffer(bytes.fromhex(row['argument_LE_F32_hex']),dtype='<f4');rs=response(line,seq,a);require(row['sequence']==seq and row['response']==line and row['result_LE_F32_hex']==rs.tobytes().hex() and row['captured_operands_used'] is False,'Original device argument/result/ordinal association differs')
 require(canonical(work['device_RS_records'])==canonical(records),'Original CPU/device argument ledger differs');owned=work['first_RMS_arguments']+(work.get('prefix4_RMS_arguments',[]) if work['prefix4_attempted'] else []);require(len(owned)==len(records),'Exact original model own RMS coverage differs')
 roles=['blk.0.hc_attn_']+([f'blk.{layer}.hc_{kind}_' for position in range(4) for layer in range(48) for kind in ('attn','ffn')]+['output_hc_'] if work['prefix4_attempted'] else []);require([r['role'] for r in records]==roles,'Exact first/prefix4 layer/HC role order differs')
 for row,device in zip(owned,records):
  a=np.asarray([v['rsqrt_argument_f32'] for v in row['source_FMA_XOR_arguments']],dtype='<f4');require(row['device_sequence']==device['sequence'] and row['role']==device['role'] and row['argument_LE_F32_hex']==device['argument_LE_F32_hex']==a.tobytes().hex() and row['captured_operand_used'] is False,'Actual own FMA/XOR argument->device correspondence differs')
 if work_root is not None:
  import owned_hc35_f32_block_control_v2 as hc
  from independent_q8_1_activation_v1 import encode
  work_root=Path(work_root).resolve()
  def load_array(binding,shape):
   return read_bound_array(binding,shape,work_root)
  matrices=[load_array(work['first_arrays']['residual_owned'],(4,2560))]
  if work['prefix4_attempted']:
   for position in range(4):
    for layer in range(48):
     for phase_name in ('input','attention'):matrices.append(load_array(work['prefix4']['arrays'][f'p{position}_l{layer}_{phase_name}'],(4,2560)))
   matrices.append(load_array(work['prefix4']['arrays']['p3_l47_ffn'],(4,2560)))
  for matrix,row in zip(matrices,owned):
   expected=[hc.rms_argument(v,hc.EPSILON) for v in matrix];require(canonical(expected)==canonical(row['source_FMA_XOR_arguments']) and hashlib.sha256(matrix.tobytes()).hexdigest()==row['residual_sha256'],'Own RMS must recompute from saved independent residuals')
  for binding in list(work['first_arrays'].values())+list(work.get('prefix4',{}).get('arrays',{}).values()):
   path=Path(binding['path']);require(path.resolve().parent==work_root and not path.is_symlink() and sha(path)==binding['sha256'] and path.stat().st_size==binding['bytes'],'Original complete own/native-target array SHA/extent changed')
  normalized=load_array(work['first_arrays']['candidate_normalized'],(4,2560));norm=load_array(work['first_arrays']['norm_owned'],(4,2560));rs=np.frombuffer(bytes.fromhex(records[0]['result_LE_F32_hex']),dtype='<f4');expected=np.asarray(np.asarray(matrices[0]*norm,dtype='<f4')*rs[:,None],dtype='<f4');require(expected.tobytes()==normalized.tobytes(),'Normalized result must use own residual/norm and actual device RS, without correction');mixed=load_array(work['first_arrays']['candidate_mixed'],(2560,));packet=encode(mixed.tobytes(),1,2560);require(sha(work_root/'first-candidate-mixed.q81')==work['first_gate']['packet_sha256'] and (work_root/'first-candidate-mixed.q81').read_bytes()==packet and hashlib.sha256(normalized.tobytes()).hexdigest()==work['first_gate']['normalized_sha256'],'Original normalized/complete packet source differs')
 require(not work['prefix4_attempted'] or work['first_gate']['passed'] is True,'Prefix4 forbidden before exact first HC gate');return {'exact_owned_CPU_device_RS_ledger_qualified':True,'request_count':len(records),'internal_HC_argument_observed':False,'full_model_math_qualified':False}
