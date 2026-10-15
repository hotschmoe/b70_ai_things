from pathlib import Path
import hashlib,json,struct,re,collections
P=Path(__file__).resolve().parent;B=P.parent/'build';sha=lambda d:hashlib.sha256(d).hexdigest()
records={}
for line in(B/'sha256.txt').read_text().splitlines():h,n=line.split(maxsplit=1);records[n]=h
exe=(B/'attn_score_probe').read_bytes();assert sha(exe)==records['attn_score_probe']
def elf(data,base):
 h=data[base:base+64];assert h[:6]==b'\x7fELF\x02\x01';so=struct.unpack_from('<Q',h,40)[0];se,sn,si=struct.unpack_from('<HHH',h,58);ss=[struct.unpack_from('<IIQQQQIIQQ',data,base+so+k*se)for k in range(sn)];ns=ss[si];names=data[base+ns[4]:base+ns[4]+ns[5]];return{names[x[0]:].split(b'\0')[0].decode():x for x in ss},max(so+se*sn,max(x[4]+x[5]for x in ss if x[1]!=8))
selected={};tables={};bundles={}
for arm,filename in [('public','public.images.o'),('private','private.images.o')]:
 data=(B/filename).read_bytes();assert sha(data)==records[filename];top,_=elf(data,0);z=top['__CLANG_OFFLOAD_BUNDLE__sycl-spir64_gen'];off=z[4];end=off+z[5];bundle=data[off:end];bundles[arm]={'wrapper_sha256':sha(data),'bundle_bytes':len(bundle),'bundle_sha256':sha(bundle),'exact_bundle_occurrences_in_final':exe.count(bundle)};assert exe.count(bundle)==1
 counts=collections.Counter();index=0
 while off<end:
  sections,size=elf(data,off);z=sections.get('.ze_info');info=data[off+z[4]:off+z[4]+z[5]].decode()if z else'';names=re.findall(r'^  - name:\s+(\S+)',info.split('kernels_misc_info:')[0],re.M);counts.update(names)
  hits=[name for name in names if'attn_chunk_kernel_2ad0a4'in name and'dpct_kernel_scalarILi1EE'in name]
  if hits:
   assert len(hits)==1 and arm not in selected,(arm,hits);name=hits[0];image=data[off:off+size];sec=sections['.text.'+name];text=image[sec[4]:sec[4]+sec[5]];z=sections['.misc.buildOptions'];options=image[z[4]:z[4]+z[5]].decode().strip('\0');assert options=='-cl-fp32-correctly-rounded-divide-sqrt'
   (P/(arm+'.zebin')).write_bytes(image);(P/(arm+'.ze_info')).write_text(info,encoding='ascii')
   selected[arm]={'wrapper_image_index':index,'kernel':name,'image_bytes':len(image),'image_sha256':sha(image),'text_bytes':len(text),'text_sha256':sha(text),'build_options':options,'whole_image_occurrences_in_final':exe.count(image),'disasm_section_filename_bytes':len(('.text.'+name+'.asm').encode())}
  image_end=off+size;off=end if image_end==end else(image_end+15)//16*16;index+=1
 assert off==end;tables[arm]={'images':index,'kernel_IDs':dict(counts)}
assert set(selected)=={'public','private'}
public=(P.parent/'frozen/qsa_decode_attn.dp.cpp.o').read_bytes();actual=(B/'public-current-member.o').read_bytes();after=(B/'public-after.o').read_bytes();assert sha(public)=='bbae92dcf61427f35078d498a240b1c9f23862e8bf13f7b63d95252e3b06baf2'and actual==public==after
common=set(tables['public']['kernel_IDs'])&set(tables['private']['kernel_IDs'])
v={'status':'ACTUAL_WRAPPER_BINDING_SELECTED_TARGET_IDS_AUDITED','binary_sha256':sha(exe),'public_fatobject_sha256':sha(public),'public_frozen_current_after_exact':True,'bundles':bundles,'selected':selected,'target_IDs_distinct':selected['public']['kernel']!=selected['private']['kernel'],'all_overlapping_kernel_IDs':sorted(common),'tables':tables,'scope':'Actual linked bundles and static selected images only; no runtime GPU/latency/occupancy/zero-spill proof.'}
(P/'selection.json').write_text(json.dumps(v,indent=2)+'\n',encoding='ascii')
print(json.dumps({'target_IDs_distinct':v['target_IDs_distinct'],'selected':selected,'overlap_ID_count':len(common),'bundle_occurrences':bundles},indent=2))
