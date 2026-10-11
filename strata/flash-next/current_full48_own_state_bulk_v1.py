"""Fresh43/47 mode1 own-state candidate; software estimate, never native proof."""
import hashlib,json
from pathlib import Path
from types import FunctionType
from full48_owned_composition_storage_v1 import Full48OwnedComposition
import full48_owned_composition_storage_v2 as legacy
import original_native_hc_f32_v1 as scalar
from original_native_hc_bulk_f32_v1 import RowsDot32
import source43_sdk_binding_v1 as sdk43
import source47_grouped_model_binding_v1 as sdk47
require=scalar.require
SOURCE_FILES=('sycl/src/program/generate.cpp','sycl/src/core/verify.cpp','sycl/src/prefill/prefill.cpp','sycl/src/kernels/hc_native_projection.cpp','sycl/src/kernels/hc_native_composition.cpp','sycl/src/kernels/cuda/native_qsa_indexer.dp.cpp')

def source_contract(generation):
 require(type(generation)is int and generation in(43,47),'Explicit actual source43/47 required');sdk=sdk43 if generation==43 else sdk47;plan=json.loads(sdk.PLAN.read_bytes());require(hashlib.sha256(sdk.PLAN.read_bytes()).hexdigest()==sdk.PLAN_SHA,'Original actual build source proposal changed');rows={}
 for name in SOURCE_FILES:
  path=sdk.SDK/'source'/name;before=path.stat();raw=path.read_bytes();after=path.stat();sig=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns);want=plan['expected_patched_source_sha256'][name];require(not path.is_symlink()and sig(before)==sig(after)and hashlib.sha256(raw).hexdigest()==want,'Actual consumed source changed '+name);rows[name]=want
 return {'generation':generation,'source_root':str(sdk.SDK/'source'),'build_proposal_sha256':sdk.PLAN_SHA,'consumed_source_sha256':rows,'current_source_bytes_checked':True,'current_SDK_ABI_or_GPU_qualification':False,'old_source35_current_gate_passed':False}

def schedule(token_ids,args,env):
 ids=list(token_ids);require(len(ids)in(1,2,4,8)and all(type(i)is int and 0<=i<248320 for i in ids),'Bounded own original prefix1/2/4/8 required');args=list(args);env=dict(env)
 valued={'--pack','--native','--ple-gguf','--expert-profile','--expert-cache','--prefill','--spec','--max-context','--kv','--ple-io','--ple-row-cache','--pcie-frac','--vram-reserve-mib','--prompt-cache','--suffix-draft','--lookup-chain','--adapt-swaps','--adapt-every','--batch','--short-read','--conversation-cache-mib','--layer-split','--split-device'};flags={'--stream-experts','--no-prefill-borrow','--trim-stage-weights','--spec-split','--no-spec-split'};seen=set();i=0
 while i<len(args):
  key=args[i];require(key not in seen and key in valued|flags,'Unknown/duplicate current own-state option '+key);seen.add(key)
  if key in valued:require(i+1<len(args)and not args[i+1].startswith('--'),'Missing actual option value');i+=2
  else:i+=1
 expected={'--prefill':'64','--spec':'2','--max-context':'2048','--kv':'fp16','--ple-io':'mmap','--ple-row-cache':'65536','--pcie-frac':'1','--vram-reserve-mib':'2048','--prompt-cache':'0','--suffix-draft':'0','--lookup-chain':'0','--adapt-swaps':'0','--adapt-every':'0','--batch':'0'}
 for key,value in expected.items():require(legacy.option(args,key)==value,'Current exact geometry/cache0 differs '+key)
 require({'--stream-experts','--no-prefill-borrow'}<=seen and legacy.option(args,'--short-read','64')=='64'and legacy.option(args,'--conversation-cache-mib','0')=='0','Exact current fresh verifier-only schedule required')
 if '--layer-split'in args:require(legacy.option(args,'--layer-split')=='32'and legacy.option(args,'--split-device')=='1'and '--trim-stage-weights'in args,'Only declared32/16 topology supported')
 else:require('--split-device'not in args and '--trim-stage-weights'not in args,'Foreign topology options refused')
 require(not({'--spec-split','--no-spec-split'}<=seen),'Ambiguous split-window mode')
 allowed={'STRATA_SYCL_NATIVE_HC','STRATA_HC_Q8','STRATA_HC_Q8_INJECT','STRATA_QFUSE','STRATA_GR_V3','STRATA_VERIFY_DEVICE_PLAN','STRATA_VERIFY_NO_HOST','STRATA_STAGE_MIRRORS','STRATA_STAGE_MIRROR_SEGMENT_MIB','STRATA_MIRROR_MIB','STRATA_DEBUG','STRATA_REQUEST_LINES','STRATA_DEC_BATCH','STRATA_PREFILL_MMQ','STRATA_PF_FUSED','STRATA_PREFILL_BF16X2','STRATA_BF16_TC','STRATA_GDN_HEAD','STRATA_GDN_KEYHEAD','STRATA_SYCL_NATIVE_GROUP_F16','STRATA_QSA_INDEXER_MATERIALIZED_HALF'}
 require(not any(k.startswith('STRATA_')and k not in allowed for k in env),'Unreviewed arithmetic/observer environment refused; pass exact baseline config')
 for key,value in {'STRATA_SYCL_NATIVE_HC':'1','STRATA_VERIFY_DEVICE_PLAN':'1','STRATA_VERIFY_NO_HOST':'1','STRATA_STAGE_MIRRORS':'1','STRATA_STAGE_MIRROR_SEGMENT_MIB':'1024','STRATA_MIRROR_MIB':'65536'}.items():require(env.get(key)==value,'Current original native/mirror flag differs '+key)
 for key in ('STRATA_HC_Q8','STRATA_HC_Q8_INJECT','STRATA_QFUSE','STRATA_GR_V3','STRATA_PREFILL_MMQ','STRATA_PF_FUSED','STRATA_PREFILL_BF16X2','STRATA_BF16_TC','STRATA_GDN_HEAD','STRATA_GDN_KEYHEAD','STRATA_SYCL_NATIVE_GROUP_F16'):require(env.get(key,'0')=='0','Unmodeled arithmetic branch refused '+key)
 require(env.get('STRATA_DEC_BATCH','1')=='1','Changed verifier row grouping refused');require(env.get('STRATA_QSA_INDEXER_MATERIALIZED_HALF')=='1','Mode0 inline half roundtrip actual compiler semantics unresolved; executable reference is mode1 only')
 windows=[];p=0
 while p<len(ids)-1:
  n=min(2,len(ids)-1-p);windows.append({'position':p,'rows':n,'normal_dispatch':'prompt_verify','math_route':'verifier','groups':2 if '--spec-split'in args and n>=2 else 1});p+=n
 windows.append({'position':len(ids)-1,'rows':1,'normal_dispatch':'target_verify','math_route':'verifier','groups':1})
 return {'accepted_ids':ids,'windows':windows,'fresh_zero_state_required':True,'route_inputs':'accepted original IDs and explicit current config only','native_route_metadata_used':False,'native_window_group_rounding_emulated':False,'full_model_math_qualified':False}

def first_HC_gate(path,expected_report_sha256,expected_ELF_sha256):
 path=Path(path);require(path.is_file()and not path.is_symlink()and path.stat().st_size<=1<<20,'Original completed firstHC report required');raw=path.read_bytes();require(hashlib.sha256(raw).hexdigest()==expected_report_sha256,'Exact actual firstHC report SHA required');report=json.loads(raw)
 expected={'residual_input.f32','attn_hc_normalized.f32','attn_hc_low.f32','attn_hc_gate.f32','attn_hc_inject.f32','attn_mixed.f32','own-first-packet.q81'};proof=report.get('bulk_vs_executed_scalar',{});checks=proof.get('exact_owned_field_and_packet_comparisons',{})
 require(report.get('passed')is True and report.get('errors')==[]and report.get('postCPU_full4',{}).get('passed')is True and proof.get('all_bitwise_equal')is True and set(checks)==expected and all(v is True for v in checks.values())and report.get('explicit_bulk_ELF_sha256')==expected_ELF_sha256,'Actual seven-output parity plus completed fresh closure required')
 for name in expected:
  p=path.parent/name;require(p.is_file()and not p.is_symlink()and hashlib.sha256(p.read_bytes()).hexdigest()==report['artifacts'][name],'Actual firstHC owned output bytes changed')
 return {'first_HC_report_sha256':expected_report_sha256,'CPU_transport_parity_only':True,'current_model_or_GPU_authority_transferred':False}

class CurrentFull48OwnStateBulk(Full48OwnedComposition):
 def __init__(self,provider,identity,args,env,*,generation,rows_dot_elf,rows_dot_sha256,first_HC_report,first_HC_report_sha256,tile_bytes=64<<20):
  self.first_gate=first_HC_gate(first_HC_report,first_HC_report_sha256,rows_dot_sha256)
  self.generation=generation;self.args=tuple(args);self.env=tuple(sorted(dict(env).items()));self.source_binding=source_contract(generation);self.source_root=Path(self.source_binding['source_root']);schedule([248045],self.args,dict(self.env));self.rows_backend=RowsDot32(rows_dot_elf,rows_dot_sha256)
  math={k:dict(self.env)[k]for k in legacy.MATH_KEYS if k in dict(self.env)};super().__init__(provider,identity,tile_bytes=tile_bytes,runtime_options=math)
 def hc_read(self,residual,stem,inject=True):return self.rows_backend.hc_read_original(self.projector,residual,stem,self.g.eps,inject)
 def hc_write(self,residual,block,inject):return scalar.hc_write_original(residual,block,inject)
 def tokens(self,token_ids):
  self.rows_backend.recheck();namespace={**vars(legacy),'derive_schedule':schedule,'bind_dispatch_source':lambda root:source_contract(self.generation)};method=FunctionType(legacy.RouteAwareFull48OwnedComposition.tokens.__code__,namespace);result=method(self,token_ids);self.rows_backend.recheck();require(source_contract(self.generation)==self.source_binding,'Actual consumed source changed during own-state math');result.update(lane='current43_47_mode1_original_full48_own_state_bulk_candidate_v1',materialized_mode=1,mode0_actual_storage_semantics_qualified=False,grouped_F16_modeled=False,explicit_CPU_ELF=str(self.rows_backend.path),explicit_CPU_ELF_sha256=self.rows_backend.expected,old_source35_runtime_proof_transferred=False,non_HC_reference_math_changed=False,host_rsqrt_expf_device_intrinsics_qualified=False,full_model_math_qualified=False);return result
