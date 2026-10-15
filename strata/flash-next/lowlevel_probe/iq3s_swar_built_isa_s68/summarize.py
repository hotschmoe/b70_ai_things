#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,re,datetime
p=Path(__file__).parent;r=p.parent;selection=json.loads((p/'selection.json').read_text())
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
old=r/'iq3s-existing-isa-s66-prep/disasm-short/.text.iq3s_c6.asm';public_lines=old.read_text().splitlines()
arms={};excerpts={}
for arm,b in [('original',0),('candidate',1)]:
 f=p/f'{arm}-disasm'/f'.text._ZTSN3s659PrivateC6ILb{b}EEE.asm';lines=f.read_text().splitlines()
 info=(p/(arm+'.ze_info')).read_text().split('  - name:            Intel_Symbol_Table_Void_Program')[0]
 def field(k):
  m=re.search(r'^\s*'+re.escape(k)+r':\s*(.+)$',info,re.M);return m.group(1)if m else None
 pred=[i+1 for i,l in enumerate(lines)if '(ne)'in l and ':ub'in l]
 dp4=[i+1 for i,l in enumerate(lines)if re.search(r'\bdp4a\s',l)]
 d32=[i+1 for i,l in enumerate(lines)if 'load.ugm.d32.a64' in l]
 data=selection['selected'][arm]
 arms[arm]={k:data[k]for k in ['index','kernel','zebin_bytes','zebin_sha256','text_bytes','text_sha256','build_options']}
 arms[arm].update(asm_path=str(f),asm_sha256=sha(f),asm_lines=len(lines),asm_bytes=f.stat().st_size,
   resource_metadata={k:field(k)for k in ['grf_count','simd_size','eu_thread_count','intel_reqd_sub_group_size','intel_reqd_workgroup_walk_order','required_work_group_size','spill_size','per_thread_scratch_size','slm_size']},
   scratch_or_spill_keys_present=bool(re.search(r'spill|scratch',info,re.I)),
   static_byte_nonzero_predicate_instructions=len(pred),byte_predicate_line_numbers=pred,
   static_DP4A_instructions=len(dp4),DP4A_line_numbers=dp4,
   exact_d32_a64_load_message_lines=d32,
   weight_reconstruction_message_lines=[164,167,168]if arm=='original'else[167,170,171],
   weight_address_binding='row + 110*kbx + 4*iqs + 2/+6 (qs), row + 110*kbx + 2*iqs + 74 (signs); unchanged source reconstruction; d8 qh separate')
 ranges=[(139,168),(220,260),(302,338),(390,411),(642,666)]if arm=='original'else[(142,171),(225,250),(285,305),(340,362),(420,443)]
 excerpts[arm]={f'{a}-{z}':'\n'.join(f'{i+1}: {lines[i]}'for i in range(a-1,z))for a,z in ranges}
 if arm=='original':
  diffs=[{'line':i+1,'public':a,'private':b}for i,(a,b)in enumerate(zip(public_lines,lines))if a!=b]
  eots=[i for i,l in enumerate(lines)if 'end of thread' in l];assert len(eots)==1
  arms[arm]['comparison_to_S66_public_ASM']={'total_lines_same':len(lines)==len(public_lines),'different_lines':diffs,'all_differences_after_EOT':all(d['line']>eots[0]+1 for d in diffs),'not_identical_text_binary':data['text_sha256']!=selection['selected']['public']['text_sha256']}
assert arms['original']['static_byte_nonzero_predicate_instructions']==32
assert arms['candidate']['static_byte_nonzero_predicate_instructions']==0
assert arms['original']['static_DP4A_instructions']==arms['candidate']['static_DP4A_instructions']==48
(p/'assembly-excerpts.json').write_text(json.dumps(excerpts,indent=2)+'\n')
pins={}
for f in [r/'iq3s-dense-oracle-s65-prep/iq3s_dense_leaf.cpp',r/'iq3s-dense-oracle-s65-prep/private_iq3s_c6.hpp',r/'iq3s-dense-oracle-s65-prep/build/sha256.txt',r/'iq3s-dense-oracle-s65-prep/source-link-bindings.json',r/'iq3s-existing-isa-s66-prep/s66-report.json',r/'iq3s-existing-isa-s66-prep/manifest.json',Path('/mnt/vm_8tb/b70/build/strata-ab/src-ep/sycl/src/kernels/cuda/iq_kernels.dp.cpp')]:pins[str(f)]=sha(f)
report={'status':'OFFLINE_EXISTING_S65_ARTIFACT_AUDIT_COMPLETE_RUNTIME_GATE_PENDING','author':'/root/ple_upstream (native Codex)',
 'exe':selection['exe'],'exe_sha256_from_actual_build':selection['exe_sha256_from_actual_build_record'],'exe_stat_unchanged_during_metadata_audit':selection['exe_stat_unchanged'],
 'compiler_image':'sha256:e41d59c0940e1bd2f6f64c9d5d2d7452b77812b0a8407fac8d67cacec5d7fdd2','compiler_binding':'actual S65 build uses DPC++2026.1.0(2026.1.0.20260617), source/header/object pins in build input manifest; target bmg-g31',
 'command':'env B70_CPU_LOCK_TIMEOUT=0 B70_AGENT=S68-offline-existing-ISA bin/cpu-run bash '+str(p/'decode.sh'),
 'CPU_run_reported_seconds':11,'offline_decoder':'ocloc disasm -file /w/{original,candidate}.zebin -device bmg-g31 -dump /w/{original,candidate}-disasm',
 'decoder_exit':{a:int((p/(a+'-disasm.exit')).read_text())for a in arms},'no_DRI_or_network':True,'new_compilation':0,'GPU_touches':0,'model_payload_reads':0,
 'initial_metadata_selector_failure':'Preserved attempt1-section-selector. It incorrectly required exactly one .text section; auxiliary Intel_Symbol_Table_Void_Program has its own section. Fixed exact kernel-name selection before any decoder execution.',
 'alias_or_ELF_modification':False,'metadata_bytes_and_three_selected_images_read':selection['metadata_and_selected_bytes_read'],
 'public_image':selection['selected']['public'],'public_canonical_zebin_matches_S66':True,
 'arms':arms,'source_workgroup_geometry':[1,4,32],'compiled_required_workgroup_geometry':'no required_work_group_size field; walk-order vector is not workgroup geometry',
 'pins':pins,'verdict':'SWAR-only candidate is emitted distinctly: byte predicate emulation removed, smaller main text, same static DP4A count and source-bound reconstruction messages. Exact public canonical image remains unchanged. No runtime dispatch/latency/traffic/occupancy or numerical claim; S65 runtime correctness/health/teardown gate remains pending at this artifact audit.',
 'limits':['All counts are static in one compiled generic-K/C6 kernel image, not dynamic executions or cycle estimates','Absent spill/scratch metadata is not proof of zero runtime usage','Private original and public text hashes differ; two disassembly null-move immediates after EOT differ, so no identical GPU binary claim','Code size and removal of predicate instructions do not establish speed or cause an eventual runtime delta','Both eligible K values use these generic compiled kernel classes; no launch/dispatch/occupancy evidence is supplied by this audit']}
(p/'s68-report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':report['status'],'public_match':True,'resources':{a:v['resource_metadata']for a,v in arms.items()},'main_text_bytes':{a:v['text_bytes']for a,v in arms.items()},'static_byte_predicates':{a:v['static_byte_nonzero_predicate_instructions']for a,v in arms.items()},'static_DP4A':{a:v['static_DP4A_instructions']for a,v in arms.items()}},indent=2))
