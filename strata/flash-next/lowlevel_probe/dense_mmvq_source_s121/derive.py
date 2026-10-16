from pathlib import Path
import csv,re,json,hashlib,collections
R=Path('/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014');SRC=Path('/mnt/vm_8tb/b70/build/strata-ab/src-ep');D=Path(__file__).parent
C=Path('/mnt/vm_8tb/github/b70_ai_things/strata/flash-next/lowlevel_probe/current_c2_profile_s95/runtime/analysis/gpu/operations.csv')
def pin(p):
 b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
rows=list(csv.DictReader(C.open()));selected=[];heads=[];pools=collections.defaultdict(lambda:{'count':0,'total_ms':0.,'CSV_rows':0});q8cols=collections.defaultdict(lambda:{'count':0,'total_ms':0.})
def fmt(s):
 for token,f in [('launch_q6k','Q6_K'),('launch_mmvq_multi_nc<21,','IQ3_S'),('WideQ4K','Q4_K'),('WideQ5K','Q5_K'),('WideIQ4XS','IQ4_XS'),('Wide32Q8','Q8_0'),('Wide32IQ4NL','IQ4_NL'),('Q20Traits','Q2_0'),('native_q2_0_mmvq','Q2_0')]:
  if token in s:return f
 raise ValueError(s)
for lineno,r in enumerate(rows,2):
 s=r['operation']
 if r['kind']!='gpu'or'#0'not in r['process']or'mmvq'not in s:continue
 f=fmt(s);grid=int(re.search(r'SIMD\d+ \{(\d+);',s)[1]);nc=re.search(r'launch_\w+<(?:[^,<>]+, )?(\d+)>',s)
 if 'launch_mmvq_multi_nc<21,'in s:nc=re.search(r'dpct_kernel_scalar<21>, dpct_kernel_scalar<(\d+)>',s)
 nc=int(nc[1])if nc else int(re.search(r'dpct_kernel_scalar<(\d+)>',s)[1])
 item={'CSV_line':lineno,'format':f,'NCOLS':nc,'grid_x':grid,'count':int(r['count']),'recorded_total_ms':float(r['total_ms']),'recorded_median_us':float(r['median_us']),'operation':s,'process':r['process'],'engine':r['engine_or_thread']}
 if f=='Q5_K'and grid==62080:heads.append(item);continue
 selected.append(item);v=pools[f];v['count']+=item['count'];v['total_ms']+=item['recorded_total_ms'];v['CSV_rows']+=1
 if f=='Q8_0':v=q8cols[nc];v['count']+=item['count'];v['total_ms']+=item['recorded_total_ms']
selected.sort(key=lambda r:-r['recorded_total_ms']);q6=json.loads((R/'q6-dense-packed-s41-prep/q6-header-index.json').read_text());wide=json.loads((R/'primary-widek-next-s57-prep/header-projections.json').read_text());iq3=json.loads((R/'iq3s-dense-next-s63-prep/iq3s-tensor-descriptors.json').read_text())
meta=[{'format':'Q6_K','N':t['shape'][0],'M':t['shape'][1],'tensor':t['name']}for t in q6['tensors']]+wide['rows']+[dict(t,format='IQ3_S')for t in iq3['rows']]
for row in selected[:12]:
 if row['format']in['Q6_K','Q4_K','Q5_K','IQ4_XS','IQ3_S']and row['NCOLS']==6:
  possible=[t for t in meta if t['format']==row['format']and t['M']==row['grid_x']*4];row['retained_header_role_set']=[{'N':k[0],'M':k[1],'tensor_suffix':k[2],'count':n}for k,n in collections.Counter((t['N'],t['M'],re.sub(r'^blk\.\d+\.','',t['tensor']))for t in possible).items()]
result={'status':'SOURCE_ONLY_EXISTING_CSV_AND_METADATA_ATTRIBUTION','CSV':pin(C),'filter':'kind=gpu AND process contains #0 AND operation contains mmvq; exclude WideQ5K grid62080 fullvocabulary, matching existing S95 extended family predicate','scope':'Whole collection, NOT reviewed-cycle medians, no phase/event-tensor join or wall budget. BF16 mmvf/GEMM, HC, native_gu_port/native_down_port are outside this quantized-MMVQ predicate.','canonical_rows':len(rows),'nonvocabulary_rows':len(selected),'nonvocabulary_calls':sum(x['count']for x in selected),'nonvocabulary_recorded_total_ms':sum(x['recorded_total_ms']for x in selected),'format_pools':dict(sorted(pools.items(),key=lambda x:-x[1]['total_ms'])),'top12_operation_rows':selected[:12],'head_context_separate_excluded_from_nonvocab_ranking':heads,'Q8_0_columns':q8cols,'head_encoded_bytes_source_conditioned':176*(2560//256)*248320,'new_model_reads':0,'GPU_build_decode_newtrace_calls':0}
(D/'ranking.json').write_text(json.dumps(result,indent=2,ensure_ascii=True)+'\n')
source_ranges={'sycl/src/kernels/cuda/native_mmvq.dp.cpp':[(191,205),(1552,1640),(1870,1893),(1928,2000),(2080,2124),(2148,2159),(2178,2202),(3290,3301),(3856,3868),(3912,3926),(3940,3980)],'sycl/src/core/verify.cpp':[(1029,1031),(1051,1071),(1279,1284),(1387,1405),(1663,1667)],'sycl/src/kernels/cuda/shared_expert.dp.cpp':[(313,340),(356,358)],'sycl/src/core/mtp.cpp':[(206,212),(850,863),(878,883),(2190,2204)],'sycl/src/kernels/cuda/verify_kernels.dp.cpp':[(306,338)],'sycl/include/strata/sycl_math.hpp':[(1,18)],'sycl/include/strata/kernels/q8_1_finite.hpp':[(21,44)]}
excerpts={};pins=[pin(C)]
for rel,ranges in source_ranges.items():
 p=SRC/rel;ls=p.read_text().splitlines();pins.append(pin(p));excerpts[rel]=[{'first_line':s,'last_line':e,'text':'\n'.join(ls[s-1:e])}for s,e in ranges]
for p in [R/'q6-dense-packed-s41-prep/q6-header-index.json',R/'primary-widek-next-s57-prep/header-projections.json',R/'iq3s-dense-next-s63-prep/iq3s-tensor-descriptors.json',R/'q6-a4-main-s107-compiled-audit/facts.json',R/'primary-widek-resources-s57-20261015/resource-summary.json',R/'current-c2-profile-s95-analysis-followup/extended-mmvq-and-cold-limits.json',Path('/mnt/vm_8tb/github/b70_ai_things/docs/20261016_strata_q6_a4_serving.md'),Path('/mnt/vm_8tb/github/b70_ai_things/docs/20261015_strata_iq4xs_rpw_probe.md'),Path('/mnt/vm_8tb/github/b70_ai_things/docs/20261015_strata_iq3s_dense_swar_probe.md'),Path('/mnt/vm_8tb/github/b70_ai_things/docs/20261015_strata_iq3s_int8_dpas_probe.md'),R/'s116-post-reviews-20261016/root-post-decision.json']:pins.append(pin(p))
(D/'source-excerpts.json').write_text(json.dumps({'source_root':str(SRC),'excerpts':excerpts,'scope':'Exact UTF8 text excerpts JSON-escaped to ASCII; source-only, no compiler or runtime proof.'},indent=2,ensure_ascii=True)+'\n');(D/'pins.json').write_text(json.dumps({'inputs':pins,'source_and_small_metadata_only':True,'models_or_ELF_or_archive_or_trace_payload_reads':0,'candidate_executed':False},indent=2)+'\n')
print(json.dumps({'nonvocab_rows':len(selected),'calls':result['nonvocabulary_calls'],'total_ms':result['nonvocabulary_recorded_total_ms'],'format_pools':result['format_pools'],'head_bytes':result['head_encoded_bytes_source_conditioned']}))
