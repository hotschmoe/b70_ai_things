#!/usr/bin/env python3
"""Conditional captured-input zero-state recurrence; no weights/device exp."""
import argparse,ast,copy,hashlib
from pathlib import Path
from types import FunctionType
import numpy as np
import audit_source43_q81_boundaries_v1 as original
require=original.require;HERE=original.HERE
PLAN=HERE/'source43-zero-gdn-state-audit-source-plan-v1.json'
FIELDS={'gdn_normalized_qkv':40960,'gdn_decay_beta':384,'gdn_state_before':3145728,'gdn_state_after':3145728}
def state_comparison(normalized,ab,before,after,S=128,HK=16,HV=48):
 require(all(type(v)is int and v>0 for v in(S,HK,HV))and HV%HK==0,'Typed GDN geometry required');channels=2*HK*S+HV*S;count=S*HV*S
 require(len(normalized)==channels*4 and len(ab)==2*HV*4 and len(before)==len(after)==count*4,'Exact conditional recurrence extents required');require(not any(before),'Actual before-state must be bytezero, including sign bits')
 h=np.frombuffer(normalized,dtype='<f4');a=np.frombuffer(ab,dtype='<f4').reshape(2,HV);native=np.frombuffer(after,dtype='<f4').reshape(S,HV,S)
 require(np.isfinite(h).all()and np.isfinite(a).all()and np.isfinite(native).all()and np.all(a[0]<=0)and np.all((a[1]>=0)&(a[1]<=1)),'Finite zero-state source decay/beta range required')
 keys=h[HK*S:2*HK*S].reshape(HK,S);values=h[2*HK*S:].reshape(HV,S)
 # Source: qh=head%HK; zero old state gives kv=+0. For finite nonnegative
 # exp(decay), delta=F32((v-+0)*beta). The explicit FMA of g*+0 with
 # already-rounded k*delta includes signed-zero cancellation, not a
 # mathematical outer product with unconstrained precision.
 delta=np.multiply(np.subtract(values,np.float32(0),dtype=np.float32),a[1,:,None],dtype=np.float32)
 product=np.multiply(keys[np.arange(HV)%HK].T[:,:,None],delta[None,:,:],dtype=np.float32)
 expected=np.add(np.zeros((S,HV,S),dtype=np.float32),product,dtype=np.float32).astype('<f4');require(np.isfinite(delta).all()and np.isfinite(expected).all(),'Finite independently computed recurrence required');words=expected.view('<u4')!=native.view('<u4');different=np.flatnonzero(words);first=int(different[0])if len(different)else None
 return {'state_exact':len(different)==0,'different_words':int(len(different)),'first_differing_index_i_head_j':None if first is None else [first//(HV*S),(first//S)%HV,first%S],'expected_state_sha256':hashlib.sha256(expected.tobytes()).hexdigest(),'native_state_sha256':hashlib.sha256(after).hexdigest(),'input_sha256':{'normalized_qkv':hashlib.sha256(normalized).hexdigest(),'decay_beta':hashlib.sha256(ab).hexdigest(),'before_state':hashlib.sha256(before).hexdigest()},'S':S,'key_heads':HK,'value_heads':HV,'q_used_for_state':False,'actual_before_bytezero':True,'device_exp_or_rs_called':False,'captured_native_inputs_used_for_local_diagnostic':True,'original_reference_replaced':False,'full_model_math_qualified':False,'subnormal_or_fastmath_equivalence_qualified':False,'signed_zero_product_words':int(np.count_nonzero(product.view('<u4')==0x80000000)),'expected_negative_zero_words':int(np.count_nonzero(expected.view('<u4')==0x80000000))}
def ports():
 namespace={**vars(original),'PLAN':PLAN,'FIELDS':FIELDS,'state_comparison':state_comparison}
 node=copy.deepcopy(next(n for n in ast.parse(Path(original.__file__).read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='audit'))
 loops=[n for n in ast.walk(node)if isinstance(n,ast.For)and ast.unparse(n.target)=='(raw_name, packet_name, cols)'];require(len(loops)==1,'Exact inherited selected-field loop required');old=loops[0]
 replacement=ast.parse("values={}\nfor name,size in FIELDS.items():\n f=byname[name];path=Path(f['path']);require(path.parent==capture/'layer0' and f['provenance']=='actual_buffer' and report['artifacts'][str(path.relative_to(root))]==f['sha256'],'Original selected state field ownership/artifact required');values[name]=consume(path,f['sha256'],size,size)\nchecks=[state_comparison(values['gdn_normalized_qkv'],values['gdn_decay_beta'],values['gdn_state_before'],values['gdn_state_after'])]").body
 class Replace(ast.NodeTransformer):
  def visit_For(self,n):return replacement if n is old else self.generic_visit(n)
  def visit_Constant(self,n):
   if n.value=='all_eight_packets_exact':return ast.copy_location(ast.Constant('all_four_states_exact'),n)
   if n.value=='packet_exact':return ast.copy_location(ast.Constant('state_exact'),n)
   return n
 node=Replace().visit(node);exec(compile(ast.fix_missing_locations(ast.Module([node],type_ignores=[])),'[current43-zero-state-captured-field-port]','exec'),namespace)
 namespace['source_binding']=FunctionType(original.source_binding.__code__,namespace)
 namespace['run']=FunctionType(original.run.__code__,namespace,argdefs=original.run.__defaults__)
 return namespace
_PORTS=ports();run=_PORTS['run']
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--actor-root',type=Path,required=True);p.add_argument('--expected-actor-report-sha256',required=True);p.add_argument('--sealed-reference-report',type=Path,required=True);p.add_argument('--expected-reference-report-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();require(__debug__,'Ordinary Python required');run(a.actor_root,a.expected_actor_report_sha256,a.sealed_reference_report,a.expected_reference_report_sha256,a.output)
