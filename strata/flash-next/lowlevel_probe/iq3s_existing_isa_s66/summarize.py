from pathlib import Path
import struct,json,hashlib
P=Path(__file__).resolve().parent; R=P.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
sel=json.loads((P/'selection.json').read_text());al=json.loads((P/'alias-proof.json').read_text());idx=json.loads((P/'candidates.json').read_text())
def names(b):
 so=struct.unpack_from('<Q',b,40)[0]; se,sn,si=struct.unpack_from('<HHH',b,58);ss=[struct.unpack_from('<IIQQQQIIQQ',b,so+i*se) for i in range(sn)];s=ss[si];ns=b[s[4]:s[4]+s[5]]
 return [ns[s[0]:].split(b'\0')[0].decode() for s in ss]
a=names((P/'iq3s-C6.zebin').read_bytes());b=names((P/'iq3s-C6-shortname.zebin').read_bytes()); changes=[{'index':i,'before':x,'after':y} for i,(x,y) in enumerate(zip(a,b)) if x!=y]
o=al['changed_offsets'];ranges=[]
for x in o:
 if ranges and x==ranges[-1][1]:ranges[-1][1]=x+1
 else:ranges.append([x,x+1])
src=Path('/mnt/vm_8tb/b70/build/strata-ab/src-ep/sycl/src/kernels/cuda/iq_kernels.dp.cpp'); expected='acaa7da63733af9f3831b3a74b99e6176bf46a06e498e0a33c0591f7f4a29851';assert sha(src)==expected
asm=P/'disasm-short/.text.iq3s_c6.asm'; lines=asm.read_text().splitlines();assert len(lines)==1152
excerpts={f'{a}-{b}':'\n'.join(f'{i+1}: {lines[i]}' for i in range(a-1,b)) for a,b in [(139,168),(220,260),(302,338),(390,411),(536,545)]}
(P/'assembly-excerpts.json').write_text(json.dumps(excerpts,indent=2)+'\n')
q=Path(idx['exe']);st=q.stat();assert st.st_size==idx['stat']['size'] and st.st_mtime_ns==idx['stat']['mtime_ns']
report={
 'status':'EXISTING_BASELINE_ASSEMBLY_AUDIT_COMPLETE_NO_GPU_NO_NEW_COMPILE',
 'author':'/root/strata_ep_s27 (native Codex)',
 'config':{'exe':str(q),'exe_sha256_from_frozen_root_build':idx['expected_sha256'],'exe_stat_unchanged':True,'common_wrapper_sha256':'8f5a7efea1c815b497edfaa1bd33e02d86d4cf9ceea38fd9190d44bacfd7cd21','kernel_archive_sha256':'f2b53eb482814b2c9a046e524a1f76dee717fddc6fc45073a492880bb153e9a8','iq_source':str(src),'iq_source_sha256':expected,'source_hash_recomputed':True,'image':sel,'compiler_device_options':(P/'disasm-short/.misc.buildOptions').read_text().strip('\x00'),'tool_image':'sha256:e41d59c0940e1bd2f6f64c9d5d2d7452b77812b0a8407fac8d67cacec5d7fdd2'},
 'command':{'extract':'B70_AGENT=strata_ep_s27-s66 B70_CPU_LOCK_TIMEOUT=1 bin/cpu-run python3 '+str(P/'extract.py'),'disassemble':'See disasm.log and disasm-short.log: cpu-run docker run --rm --network none -u 1000:1000, stage-only bind, pinned image, source setvars, ocloc disasm -file /w/iq3s-C6[-shortname].zebin -device bmg-g31 -dump /w/disasm[-short]','no_dri':True,'tool_preflight_exit':0,'initial_disasm_exit':0,'alias_disasm_exit':0},
 'result':{'metadata_images_indexed':idx['images'],'metadata_bytes_read':idx['metadata_bytes_read'],'one_main_kernel_disassembled':True,'asm_bytes':asm.stat().st_size,'asm_lines':len(lines),'asm_sha256':sha(asm),'main_text_sha256':al['original_and_aliased_main_text_sha256'],'weight_reconstruction':{'source':'Split21::load qs two get_int_b2 plus signs one get_int_b2; each source helper uses two16-bit values','emitted':'three load.ugm.d32.a64 for qs/signs; no six d16 messages','asm_lines':[164,167,168],'address_proof':'loop block address is row+110*kbx; iqs=2*(k%8). qs two d32 addresses row+110*kbx+4*iqs+2 and +6; signs d32 row+110*kbx+2*iqs+74. qh byte is distinct at +66+iqs/2. Source qs[iqs] uses four-byte get_int_b2 indexing.'},'byte_emulation':{'survives':True,'byte_nonzero_predicate_static_instructions':sum('(ne)' in l and ':ub' in l for l in lines),'evidence':'per-byte mov(ne) predicate, byte selects, extract/repack; word add with negated operand implements byte subtraction after sign extension. See excerpts220-260/302-338/390-411. Not a dynamic instruction count.'}},
 'alias_recovery':{'cause':'original text section name plus .asm exceeds filesystem NAME_MAX; initial exit0 did not create main ASM','copy_only_never_runtime_or_link':True,'main_text_identical':True,'only_shstrtab_changed':al['only_shstrtab_changes'],'changed_byte_ranges_half_open':ranges,'section_name_effects':changes,'other_section_contents_identical':True,'original_canonical_preserved':True,'original_sha256':al['original_zebin_sha256'],'alias_sha256':al['aliased_zebin_sha256']},
 'verdict':'Baseline folds the six source halfword loads into three d32 loads, eliminating a six-to-three emitted-load-count rationale. Byte sign/subtract expansion remains, so existing SWAR helper substitution is a distinct bounded source candidate. No candidate image, dispatch, speed, traffic, causality, occupancy or numerical claim.',
 'provenance':{str(p.relative_to(R)):sha(p) for p in [R/'aot-reuse-s60-execution/root-build-closure.json',R/'aot-reuse-s60-execution/subagent-link-audit.json',R/'iq4xs-rpw2-s59-prep/build/sha256.txt',R/'iq3s-dense-next-s63-prep/manifest.json']}}
(P/'s66-report.json').write_text(json.dumps(report,indent=2,ensure_ascii=True)+'\n')
