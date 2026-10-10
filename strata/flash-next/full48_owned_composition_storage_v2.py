"""Owned original48 with source35 normal short-prompt route/storage schedule.
Accepted IDs/config/source only; no captured routes, activations, state or routing.
Native window/group shape rounding remains unqualified.
"""
import hashlib,json
from pathlib import Path
import numpy as np
from full48_owned_composition_storage_v1 import Full48OwnedComposition,BoundOriginalFull48Rows,UNSUPPORTED
from full48_owned_composition_storage_v1 import f32,QsaOwnedState,QsaGeometry

HERE=Path(__file__).resolve().parent
PLAN=HERE/'current-ple-prompt35-engine-build-plan-v1.json'
PLAN_SHA='82004f6cee0f975c433d245b304aff10d67dd33e028cf926a08ef5eead892ef7'
CODE=('sycl/src/program/generate.cpp','sycl/src/core/verify.cpp','sycl/src/prefill/prefill.cpp')
MATH_KEYS=('STRATA_SYCL_NATIVE_HC','STRATA_PREFILL_MMQ','STRATA_PF_FUSED','STRATA_PREFILL_BF16X2','STRATA_BF16_TC','STRATA_GDN_HEAD','STRATA_GDN_KEYHEAD','STRATA_GDN_REC_HEADS')

def require(ok,message):
 if not ok:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def bind_dispatch_source(source_root):
 require(sha(PLAN)==PLAN_SHA,'Exact source35 dispatch recipe changed')
 plan=json.loads(PLAN.read_text());root=Path(source_root).resolve();rows={}
 for name in CODE:
  path=root/name;before=path.stat();require(not path.is_symlink() and sha(path)==plan['expected_patched_source_sha256'][name],'Pinned source35 dispatch/storage code changed '+name);after=path.stat();sig=lambda s:[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns];require(sig(before)==sig(after),'Source35 code stat changed during hash');rows[name]={'path':str(path),'sha256':plan['expected_patched_source_sha256'][name],'stat5':sig(after)}
 return {'plan_sha256':PLAN_SHA,'files':rows,'source_dispatch_rule_bound':True,'actual_GPU_dispatch_witnessed':False}

def option(args,name,default=None):
 require(args.count(name)<=1,'Ambiguous duplicate source option '+name)
 if name not in args:return default
 at=args.index(name);require(at+1<len(args) and not args[at+1].startswith('--'),'Missing source option '+name);return args[at+1]

def derive_schedule(token_ids,args,env):
 ids=list(token_ids);args=list(args);env=dict(env)
 require(len(ids) in (1,2,4,8) and all(type(t) is int and 0<=t<248320 for t in ids),'Bounded accepted prefix1/2/4/8 IDs required')
 valued={'--pack','--native','--ple-gguf','--expert-profile','--expert-cache','--prefill','--spec','--suffix-draft','--lookup-chain','--max-context','--kv','--ple-io','--ple-row-cache','--pcie-frac','--vram-reserve-mib','--prompt-cache','--adapt-swaps','--batch','--adapt-every','--prompt-cache-root','--prompt-cache-every','--turn-token','--short-read','--layer-split','--split-device'}
 flags={'--stream-experts','--no-prefill-borrow','--trim-stage-weights','--spec-split','--no-spec-split'}
 seen=set();at=0
 while at<len(args):
  name=args[at];require(name not in seen,'Duplicate admitted argv flag '+name);seen.add(name)
  require(name in valued or name in flags,'Unknown/unreviewed admitted argv option '+name)
  if name in valued:require(at+1<len(args) and not args[at+1].startswith('--'),'Missing admitted argv value '+name);at+=2
  else:at+=1
 require({'--stream-experts','--no-prefill-borrow'}<=seen,'Exact streamed/noBorrow pilot required')
 for name,want in {'--pack':'/pack','--native':'/model/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf','--ple-gguf':'/model/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf','--expert-profile':'/src/data/expert-profile.bin','--expert-cache':'auto','--ple-io':'mmap','--pcie-frac':'1','--vram-reserve-mib':'2048','--turn-token':'248045'}.items():require(option(args,name)==want,'Changed locked pilot argument '+name)
 if '--layer-split' in args:require(option(args,'--layer-split')=='32' and option(args,'--split-device')=='1' and '--trim-stage-weights' in args,'Only admitted source-native32/16 pair supported')
 else:require('--split-device' not in args and '--trim-stage-weights' not in args,'Foreign topology options unsupported')
 allowed_strata={'STRATA_DEBUG','STRATA_REQUEST_LINES','STRATA_SYCL_NATIVE_HC','STRATA_HC_Q8','STRATA_HC_Q8_INJECT','STRATA_QFUSE','STRATA_GR_V3','STRATA_VERIFY_DEVICE_PLAN','STRATA_VERIFY_NO_HOST','STRATA_MIRROR_MIB','STRATA_STAGE_MIRRORS','STRATA_STAGE_MIRROR_SEGMENT_MIB','STRATA_BATCH_FULL_STATE_CHAIN','STRATA_BATCH_PUBLIC_PREFIX','STRATA_CACHE_MESSAGE_BOUNDARY','STRATA_PREFILL_MMQ','STRATA_PF_FUSED','STRATA_PREFILL_BF16X2','STRATA_BF16_TC','STRATA_GDN_HEAD','STRATA_GDN_KEYHEAD','STRATA_GDN_REC_HEADS','STRATA_WMMA_GEMM','STRATA_DEC_BATCH'}
 require(not any(name.startswith('STRATA_') and name not in allowed_strata for name in env),'Unknown/unreviewed STRATA source math/cache flag')
 for name,want in {'STRATA_SYCL_NATIVE_HC':'1','STRATA_VERIFY_DEVICE_PLAN':'1','STRATA_VERIFY_NO_HOST':'1','STRATA_STAGE_MIRRORS':'1','STRATA_STAGE_MIRROR_SEGMENT_MIB':'1024','STRATA_MIRROR_MIB':'65536'}.items():require(env.get(name)==want,'Changed admitted native/mirror source flag '+name)
 require(env.get('STRATA_DEC_BATCH','1')=='1','Changed verifier batch storage profile')
 required={'--spec':'2','--max-context':'2048','--prefill':'64','--ple-row-cache':'65536','--kv':'fp16','--suffix-draft':'0','--lookup-chain':'0','--batch':'0','--adapt-every':'0','--adapt-swaps':'0','--prompt-cache':'3','--prompt-cache-root':'1','--prompt-cache-every':'1000000000'}
 for name,want in required.items():require(option(args,name)==want,'Unsupported/changed admitted pilot source config '+name)
 require(option(args,'--short-read','64')=='64','Changed short_read route profile unsupported')
 require(option(args,'--conversation-cache-mib','0')=='0','Parked/cached incoming state unsupported')
 require('--no-prefill-borrow' in args,'Matched noBorrow profile required')
 for name in ('--mtp','--pipeline-windows','--spec-oracle','--spec-follow','--spec-corrupt','--no-ple','--prefill-until','--tokens','--embeddings'):
  require(name not in args,'Unsupported source route/request option '+name)
 require(not any(name.startswith(('--control-vector','--rope-')) for name in args),'Alternative arithmetic source options unsupported')
 require(option(args,'--expert-scale','1')=='1','Alternative expert scale unsupported')
 for name in ('STRATA_CKPT_REREAD','STRATA_PIPELINE_SWITCH','STRATA_VERIFY_EAGER','STRATA_GDN_REC_HEADS'):
  require(name not in env,'Unsupported source presence flag including0 '+name)
 for name in ('STRATA_BATCH_FULL_STATE_CHAIN','STRATA_BATCH_PUBLIC_PREFIX','STRATA_CACHE_MESSAGE_BOUNDARY','STRATA_HC_Q8','STRATA_HC_Q8_INJECT','STRATA_QFUSE','STRATA_GR_V3','STRATA_PREFILL_MMQ','STRATA_PF_FUSED','STRATA_PREFILL_BF16X2','STRATA_BF16_TC','STRATA_GDN_HEAD','STRATA_GDN_KEYHEAD','STRATA_WMMA_GEMM'):
  require(env.get(name,'0')=='0','Unsupported/changed source math/cache flag '+name)
 require(env.get('STRATA_SYCL_NATIVE_HC')=='1','Original nativeHC profile required')
 split='--spec-split' in args;require(not(split and '--no-spec-split' in args),'Ambiguous source split profile')
 turn=option(args,'--turn-token');require(turn is not None and turn.isdecimal(),'Actual tokenizer turn token required')
 require(int(turn) not in ids[1:],'Internal turn/checkpoint cut unsupported in bounded source profile')
 windows=[];p=0
 # Pinned windows_ok: absent reread + text + n-1 <= short_read64. Fresh is
 # guaranteed by caller admission, not imported state or observed route labels.
 while p<len(ids)-1:
  T=min(2,len(ids)-1-p);windows.append({'position':p,'rows':T,'normal_dispatch':'prompt_verify','math_route':'verifier','groups':2 if split and T>=2 else 1});p+=T
 windows.append({'position':len(ids)-1,'rows':1,'normal_dispatch':'target_verify','math_route':'verifier','groups':1})
 return {'accepted_ids':ids,'windows':windows,'fresh_zero_state_required':True,'route_inputs':'acceptedIDs+admittedargs/env+pinnedsource only','native_route_metadata_used':False,'native_window_group_rounding_emulated':False,'full_model_math_qualified':False}

class RouteAwareFull48OwnedComposition(Full48OwnedComposition):
 def __init__(self,provider,source_identity_sha256,args,env,source_root,tile_bytes=64<<20):
  self.args=tuple(args);self.env=tuple(sorted(dict(env).items()));self.source_root=Path(source_root).resolve();self.source_binding=bind_dispatch_source(self.source_root)
  derive_schedule([19],self.args,dict(self.env))
  math={key:dict(self.env)[key] for key in MATH_KEYS if key in dict(self.env)}
  super().__init__(provider,source_identity_sha256,tile_bytes=tile_bytes,runtime_options=math)
 def tokens(self,token_ids):
  ids=list(token_ids)
  if len(ids) not in (1,2,4,8) or any(type(token) is not int or not 0<=token<248320 for token in ids):raise ValueError('Own prefix1/2/4/8 token roster required')
  schedule=derive_schedule(ids,self.args,dict(self.env));self.source_binding=bind_dispatch_source(self.source_root)
  by_position={position:window for window in schedule['windows'] for position in range(window['position'],window['position']+window['rows'])}
  self.projector.events=[];self.ple.reset();states={layer:model.initial_state() for layer,model in self.gdn.items()};qstates={layer:QsaOwnedState(QsaGeometry(),model.vector('indexer.k_norm.weight'),max_cells=8) for layer,model in self.qsa.items()};trace=[]
  for position,token in enumerate(ids):
   window=by_position[position];route=window['math_route'];embedding=f32(self.p.rows('token_embd.weight',[token])[0]);residual=np.broadcast_to(embedding,(4,2560)).copy();layers=[]
   for layer in range(48):
    before=residual.copy();ple_info=None
    if layer==1:ple_info=self.ple.advance(token,residual);residual=ple_info['postprojection']['result']
    attention=self.hc_read(residual,'blk.%d.hc_attn_'%layer)
    if layer in self.gdn:states[layer],detail=self.gdn[layer].mixer(attention['mixed'],states[layer],route);block=detail['output']
    else:detail=self.qsa[layer].step(qstates[layer],token,attention['mixed'],route);block=detail['block_output']
    after_attn=self.hc_write(residual,block,attention['inject']);ffn_read=self.hc_read(after_attn,'blk.%d.hc_ffn_'%layer);ffn=self.ffn[layer].run(ffn_read['mixed'],route);residual=self.hc_write(after_attn,ffn['block_output'],ffn_read['inject'])
    layers.append({'layer':layer,'family':'QSA' if layer%4==3 else 'GDN','route':route,'input':before,'attention':after_attn,'ffn':residual.copy(),'selected_ids_owned':detail.get('selected_ids_owned'),'router_ids_owned':ffn['router_ids_owned'].copy(),'PLE_rows_owned':None if ple_info is None else ple_info['row_ids_owned']})
   trace.append({'position':position,'token':token,'route':route,'normal_dispatch':window['normal_dispatch'],'declared_window_position':window['position'],'declared_window_rows':window['rows'],'declared_group_count':window['groups'],'actual_window_rounding_emulated':False,'layers':layers})
  final=self.hc_read(residual,'output_hc_',inject=False);logits=self.projector.project('output.weight',final['mixed'],activation='q8_1')
  return {'lane':'owned_original48_route_storage_mathematics_unqualified_v2','declared_schedule':schedule,'source_dispatch_binding':self.source_binding,'native_routes_used_as_math_inputs':False,'actual_window_group_shape_rounding_qualified':False,'ids':ids,'trace':trace,'first_generated_logits':logits,'gdn_states_owned':states,'qsa_states_owned':qstates,'ple_history_owned':self.ple.history.copy(),'last_two_owned':list(self.ple.last_two),'packet_events':list(self.projector.events),'captured_inputs_used':False,'captured_states_or_selected_ids_used':False,'state_initialized_from_zero':True,'actual_original_payload_used':self.p.actual_source,'actual_runtime_dispatch_witnessed':False,'nativebitwise_qualified':False,'full_model_math_qualified':False,'tolerance_gate':None,'unsupported':list(UNSUPPORTED),'source_identity_sha256':self.identity}
