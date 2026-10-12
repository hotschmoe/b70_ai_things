"""Same selected Clock/Chrome joins, per-device named kernel unions, UNADMITTED."""
import argparse,ast,collections,copy,hashlib,json,math
from pathlib import Path
import inspect_source52_p2048_profile_v1 as original
require=original.require

def family(name):
 n=name.lower().replace('_','')
 if name.startswith('zeCommandListAppendMemoryCopy'):return 'explicit_DMA_copy'
 if name.startswith('zeCommandListAppendMemoryFill'):return 'memory_fill'
 if name.startswith('zeCommandListAppendBarrier'):return 'barrier'
 if 'dequant'in n:return 'dequantization'
 if '::hcnorm'in n or '::hcprojection'in n or '::hcsilu'in n or '::hcmix'in n or '::hcstandalonewrite'in n:return 'HC_native'
 if 'gemm'in n:return 'GEMM_vendor_or_named'
 if 'gdn'in n:return 'GDN_named'
 if 'qsa'in n or 'indexer'in n or 'attn'in n:return 'QSA_or_attention_named'
 if 'nativegu'in n:return 'MoE_native_gu'
 if 'nativedown'in n:return 'MoE_native_down'
 if 'swiglu'in n or 'routekernel'in n:return 'MoE_route_or_activation'
 if 'ple'in n or 'ngram'in n:return 'PLE_or_ngram_named'
 return 'other_named_operations'

def percentile(values,p):return sorted(values)[max(0,math.ceil(len(values)*p/100)-1)]if values else None
def detail(instances,devices):
 output=[]
 for dev,origin in devices.items():
  rows=[(a,b,ordinal,index,name)for d,a,b,ordinal,index,name in instances.values()if d==dev];freq=origin['timer_frequency_hz'];factor=1000/freq;all_spans=[(a,b)for a,b,*rest in rows];merged=original.union([(a,b)for a,b in all_spans if b>a]);hull=merged[-1][1]-merged[0][0];gaps=[(a[1],b[0])for a,b in zip(merged,merged[1:])if b[0]>a[1]];families={};names={};engines={}
  for a,b,ordinal,index,name in rows:
   require(type(a)is int and type(b)is int and a<=b and type(name)is str,'Actual named raw operation shape differs');families.setdefault(family(name),[]).append((a,b));short=name.split('[SIMD',1)[0];names.setdefault(short,[]).append((a,b));engines.setdefault((ordinal,index),[]).append((a,b))
  def summary(spans):
   union=original.union([(a,b)for a,b in spans if b>a]);return {'operations':len(spans),'observed_union_ms':sum(b-a for a,b in union)*factor,'union_components':len(union)}
  topgaps=sorted(gaps,key=lambda x:x[1]-x[0],reverse=True)[:10];left={a:[]for a,b in topgaps};right={b:[]for a,b in topgaps}
  for a,b,*rest in rows:
   name=rest[-1]
   if b in left and len(left[b])<3:left[b].append(name[:240])
   if a in right and len(right[a])<3:right[a].append(name[:240])
  output.append({'device_handle':dev,'descriptor_id':origin['descriptor_id'],'frequency_hz':freq,'same_device_raw_operation_count':len(rows),'all_observed_operations':summary(all_spans),'raw_hull_ms':hull*factor,'uncovered_in_observed_hull_ms':(hull-sum(b-a for a,b in merged))*factor,'observed_gap_count':len(gaps),'gap_P50_us':percentile([b-a for a,b in gaps],50)*1000000/freq if gaps else None,'gap_P95_us':percentile([b-a for a,b in gaps],95)*1000000/freq if gaps else None,'gap_P99_us':percentile([b-a for a,b in gaps],99)*1000000/freq if gaps else None,'largest_gap_context':[{'gap_ms':(b-a)*factor,'raw_start_ticks':a,'raw_end_ticks':b,'left_names':left[a],'right_names':right[b]}for a,b in topgaps],'family_unions_nonadditive':{k:summary(v)for k,v in sorted(families.items())},'top_named_kernel_unions':[{'name':n[:320],**summary(v)}for n,v in sorted(names.items(),key=lambda item:sum(b-a for a,b in original.union(item[1])),reverse=True)[:20]],'engine_unions_nonadditive':{str(k):summary(v)for k,v in engines.items()},'physical_idle_qualified':False,'prefill_decode_temporal_split_qualified':False,'named_GEMM_role_or_layer_qualified':False})
 return output

def inspect(directory,expected_prior):
 # Add only an aggregate over original exact selected instances, without altering
 # the parser, selected bounds, device/query joins or raw before/after hashes.
 node=copy.deepcopy(next(n for n in ast.parse(Path(original.__file__).read_bytes()).body if isinstance(n,ast.FunctionDef)and n.name=='inspect'));require(isinstance(node.body[-1],ast.Return),'Original inspector return changed');node.body.insert(len(node.body)-1,ast.parse('kernel_detail=detail(instances,devices)').body[0]);namespace={**vars(original),'detail':detail};exec(compile(ast.fix_missing_locations(ast.Module([node],type_ignores=[])),__file__+'[unchanged-selected-joins-plus-aggregate]','exec'),namespace)
 # Retain the locals through a separate output-only dictionary field.
 ret=node.body[-1].value;require(isinstance(ret,ast.Dict),'Original actual reduction shape changed');ret.keys.append(ast.Constant('kernel_detail'));ret.values.append(ast.Name(id='kernel_detail',ctx=ast.Load()));exec(compile(ast.fix_missing_locations(ast.Module([node],type_ignores=[])),__file__+'[original-aggregate-output]','exec'),namespace);result=namespace['inspect'](directory);base={k:v for k,v in result.items()if k!='kernel_detail'};require(base==expected_prior,'Original selected bounds/raw bindings/reduction changed since prior observed receipt');result.update(admission_status='UNADMITTED_HISTORICAL_DIAGNOSTIC',per_arm_or_experiment_reader_passed=False,prefill_decode_temporal_split_qualified=False,selected_CPU_command_gap_metrics=None,CPU_gap_scope_reason='cpu_op rows lack selected instance-ID joins; no calibrated host selection bracket is inferred',kernel_names_prove_role_or_arithmetic=False,model_or_GPU_execution_performed=False);return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--prior-observed',type=Path,required=True);p.add_argument('--expected-prior-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();require(not a.output.exists()and original.sha(a.prior_observed)==a.expected_prior_sha256,'New output and exact observed receipt required');prior=json.loads(a.prior_observed.read_bytes());result=inspect(a.directory,prior);require(original.sha(a.prior_observed)==a.expected_prior_sha256,'Prior receipt changed');result['prior_observed_sha256']=a.expected_prior_sha256;result['analysis_source_sha256']=original.sha(Path(__file__));result['original_inspector_source_sha256']=original.sha(Path(original.__file__));a.output.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='ascii')
