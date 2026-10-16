"""Root-reviewed CPU-only existing-image audit; no decoder/build/GPU."""
from pathlib import Path
import hashlib,json,re,struct,sys
import helpers
from helpers import ELF,bind,bounded_cstring,elf_at
from compare import compare_images,raw_inventory

P=Path(__file__).resolve().parent;R=P.parent
assert len(sys.argv)==2,'audit.py NEW_OUTPUT_DIRECTORY'
O=Path(sys.argv[1]).resolve();O.mkdir(exist_ok=False)
pins=json.loads((P/'pins.json').read_text());sha=lambda b:hashlib.sha256(b).hexdigest()
def stat(p):
 z=p.stat();return {'dev':z.st_dev,'ino':z.st_ino,'bytes':z.st_size,'mtime_ns':z.st_mtime_ns}
def save(name,j):(O/name).write_text(json.dumps(j,indent=2)+'\n',encoding='ascii')
for pin in pins['small_pins']:
 b=Path(pin['path']).read_bytes();assert len(b)==pin['bytes']and sha(b)==pin['sha256'],pin['path']
EXE=Path(pins['exe']['path']);OB=Path(pins['object']['path'])
assert stat(EXE)==pins['exe']['stat'] and stat(OB)==pins['object']['stat']
build=json.loads((R/'q6-a4-main-s107-build/build/root-build-audit.json').read_text())
assert build['actual_exit']==build['tee_exit']==0
assert build['build_sha256']['/out/strata-q6-a4-s107']==pins['exe']['retained_sha256']
assert build['build_sha256']['libstrata_kernels.a']==pins['archive']['retained_sha256']
assert build['build_sha256']['CMakeFiles/strata_kernels.dir/src/kernels/cuda/iq_kernels.dp.cpp.o']==pins['object']['retained_sha256']
receipt=(R/'mtp-parallel-argmax-main-s114-build/build/build-sha256.txt').read_text()
assert pins['archive']['retained_sha256']+'  libstrata_kernels.a' in receipt
assert pins['object']['retained_sha256']+'  CMakeFiles/strata_kernels.dir/src/kernels/cuda/iq_kernels.dp.cpp.o' in receipt
# Small IQ object only: NO archive walk or complete main/archive/model hash.
assert sha(OB.read_bytes())==pins['object']['retained_sha256']
prior=json.loads((R/'routed-iq2-multi-s100-main-audit/selection.json').read_text())
assert prior['IQ_object_sha256']==pins['object']['retained_sha256'] and prior['IQ_archive_member_exact']
src=Path('/mnt/vm_8tb/b70/build/strata-ab/src-ep')
iq=(src/'sycl/src/kernels/cuda/iq_kernels.dp.cpp').read_text()
gg=(src/'include/strata/artifact/gguf_reader.hpp').read_text()
assert re.search(r'case 17:\s*return "IQ2_XS";',gg)
assert re.search(r'case 16:\s*return "IQ2_XXS";',gg)
fmt=iq.split('template<> struct Fmt<17> {',1)[1].split('template<> struct Fmt<18>',1)[0]
assert 'vec_dot_iq2_xs_q8_1' in fmt and 'vec_dot_iq2_xxs_q8_1' not in fmt
assert 'qk = 256, ipb = 8, step = 2' in fmt
assert 'static constexpr bool has = false;' in iq.split('template <> struct Multi<17>',1)[1].split('template <> struct Multi<',1)[0]
public=ELF(OB);final=ELF(EXE);helpers.final=final
pred=lambda n:'native_gu_port'in n and re.search(r'dpct_kernel_scalarILi17EE.*ILi8EE.*ILi1EE',n)is not None
host=bind(public,pred)
assert host['object_info']==prior['hosts']['OFF']['object_info']
assert host['object_cache_guard']==prior['hosts']['OFF']['object_cache_guard']
name=host['registered_name_pointer']['name'];assert name==prior['selected']['OFF']['kernel']
assert len({x['offset'] for x in host['final_cache_guard']})==2
unique={}
pattern=re.compile(rb'(?m)^  - name:[ \t]+'+re.escape(name.encode())+rb'[ \t]*\r?$')
for match in pattern.finditer(final.data):
 at=match.start();base=final.data.rfind(b'\x7fELF\x02\x01',max(0,at-4*1024*1024),at)
 assert base>=0;ss,size=elf_at(base);z=ss['.ze_info'];assert base+z[4]<=at<base+z[4]+z[5]
 unique[base]=(ss,size)
assert len(unique)==1,'Exact target image uniqueness HOLD; no all-image walk'
base,(ss,size)=next(iter(unique.items()));image=final.data[base:base+size]
z=ss['.ze_info'];ze=image[z[4]:z[4]+z[5]].decode();t=ss['.text.'+name];text=image[t[4]:t[4]+t[5]]
names=re.findall(r'^  - name:\s+(\S+)',ze.split('kernels_misc_info:')[0],re.M)
assert names==[name,'Intel_Symbol_Table_Void_Program']
(O/'selected.zebin').write_bytes(image);(O/'selected.ze_info').write_text(ze,encoding='ascii')
block=ze.split('  - name: '+name,1)[1].split('  - name:',1)[0]
fields={k:re.findall(r'(?m)^\s*'+re.escape(k)+r':\s*(.+)$',block)for k in ['grf_count','simd_size','eu_thread_count','slm_size','barrier_count','spill_size','scratch_size']}
selection={'status':'EXISTING_S107_GU17_HOST_IMAGE_BOUND_COMPARISON_PENDING','host':host,
 'selected':{'kernel':name,'executable_offset':base,'image_bytes':len(image),'image_sha256':sha(image),
 'text_bytes':len(text),'text_sha256':sha(text),'resource_fields':fields,'absent_fields':'UNKNOWN, not measured zero'},
 'context':{'retained_main_sha256':pins['exe']['retained_sha256'],'main_fresh_hash':False,
 'small_object_sha256_verified':pins['object']['retained_sha256'],'archive_identity_reused':pins['archive']['retained_sha256'],
 'archive_walk_or_rehash':False,'S114_same_archive_receipt':'mtp-parallel-argmax-main-s114-build/build/build-sha256.txt',
 'type':'Fmt17 -> IQ2_XS helper; Fmt16 -> IQ2_XXS, not interchangeable',
 'LN':8,'rows_per_WG':32,'source_shape':'N2560/FF640; S95 rowgrid40 matches FF640 under source multiple32 condition. No per-replay argument trace.',
 'actual_host_ASM_reused':'routed-iq2-multi-s100-main-audit/host-function-0.asm (same IQ object SHA)',
 'all_image_walk':False,'GPU':False,'build':False,'decoder':False}}
save('selection.json',selection)
reference=R/'routed-iq2-existing-isa-s97/qt17.zebin'
save('raw-section-inventory.json',raw_inventory(reference,O/'selected.zebin',name))
try:
 comparison=compare_images(reference,O/'selected.zebin','37cb80cd3034716da7cb4916cbe2379a35abc375169c171568f8525fe780cd78',name,7296)
 assert fields['grf_count']==['128']and fields['simd_size']==['32']and fields['eu_thread_count']==['8']
except Exception as error:
 save('HOLD.json',{'status':'HOLD_ROOT_SELECTED_INSPECTION_NO_AUTOMATIC_DECODER','reason':str(error),'selected':selection['selected']})
 raise
save('comparison.json',comparison)
# Existing S97 native prefix is exact: reuse its one selected decoder output.
asm=R/'routed-iq2-existing-isa-s97/disasm/.text.s97_qt17.asm'
ab=asm.read_bytes();assert sha(ab)=='4a75d3931dc6c810601bcf844ac81101d6c646c725ce84789e42ad697234339a'
(O/'retained-K-loop-read-window.txt').write_text('S97 selected native ASM lines190..480; raw operands, roles require root/source reading.\n'+''.join(ab.decode().splitlines(keepends=True)[189:480]),encoding='ascii')
save('isa-reuse.json',{'status':'PASS_S107_GU17_NATIVE_PREFIX_AND_METADATA_REUSE_S97_NO_DECODER_NEEDED',
 'comparison':comparison,'actual_prior_ASM':str(asm),'ASM_sha256':'4a75d3931dc6c810601bcf844ac81101d6c646c725ce84789e42ad697234339a',
 'next_read':'Map each activation word/half-scale send and address provenance within K loop against IQ2_XS helper; no opcode count -> traffic/saving claim.',
 'limits':['TG17 only23.18% of whole-collection GU service, not representative dominant path.','TG21 IQ3_S26.39% requires preserving existing E4/E2/tail weight reuse; source/compiled audit separate before any pivot.','SLM/spill/scratch absent UNKNOWN; static counters not dynamic costs.','Raw full images/text and embedded SPIR-V equality are not implied by prefix reuse.','No numerical fixture, speed, runtime dispatch or kernel authorization.']})
assert stat(EXE)==pins['exe']['stat'] and stat(OB)==pins['object']['stat']
print('PASS_EXISTING_GU17_BINDING_AND_S97_ISA_REUSE_NO_NEW_DECODER')
