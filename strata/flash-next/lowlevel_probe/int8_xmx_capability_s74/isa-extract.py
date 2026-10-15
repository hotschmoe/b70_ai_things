#!/usr/bin/env python3
# S68 seek-based ELF inventory, narrowed to the single S74 capability kernel.
from pathlib import Path
import hashlib,json,struct,time
p=Path(__file__).resolve().parent
exe=p.parent/'int8-xmx-capability-s74-prep/build-s74/int8_xmx_capability_s74'
expected='37728fb9a58097420c43f2b383574a209c154ce4d1420a90c4182326795eb9ff'
sha=lambda b:hashlib.sha256(b).hexdigest()
t0=time.monotonic();stat=exe.stat();actual=sha(exe.read_bytes());assert actual==expected
readbytes=0
with exe.open('rb') as f:
 def rd(off,n):
  global readbytes
  f.seek(off);data=f.read(n);assert len(data)==n;readbytes+=n;return data
 def elf(base):
  h=rd(base,64);assert h[:6]==b'\x7fELF\x02\x01'
  so=struct.unpack_from('<Q',h,40)[0];se,sn,si=struct.unpack_from('<HHH',h,58)
  raw=rd(base+so,se*sn);ss=[struct.unpack_from('<IIQQQQIIQQ',raw,i*se) for i in range(sn)]
  ns=ss[si];names=rd(base+ns[4],ns[5]);sections={names[s[0]:].split(b'\0')[0].decode():s for s in ss}
  size=max(so+se*sn,max(s[4]+s[5] for s in ss if s[1]!=8))
  return sections,size
 top,_=elf(0);bundle=top['__CLANG_OFFLOAD_BUNDLE__sycl-spir64_gen'];off=bundle[4];end=off+bundle[5];index=0;selected=[]
 while off<end:
  ss,size=elf(off);z=ss.get('.ze_info');info=rd(off+z[4],z[5]).decode() if z else ''
  if 'Int8K32Capability' in info:
   data=rd(off,size);kernel_sections={n:s for n,s in ss.items() if n.startswith('.text.') and 'Int8K32Capability' in n}
   assert len(kernel_sections)==1
   name,s=next(iter(kernel_sections.items()));text=data[s[4]:s[4]+s[5]]
   op=ss.get('.misc.buildOptions');options=rd(off+op[4],op[5]).decode().strip('\0') if op else None
   (p/'capability.zebin').write_bytes(data);(p/'capability.ze_info').write_text(info);(p/'main-text.bin').write_bytes(text)
   selected.append({'index':index,'offset':off,'zebin_bytes':size,'zebin_sha256':sha(data),'kernel':name.removeprefix('.text.'),'text_section':name,'text_bytes':len(text),'text_sha256':sha(text),'build_options':options,'resource_metadata':info})
  off=(off+size+15)//16*16;index+=1
 assert len(selected)==1
assert exe.stat().st_size==stat.st_size and exe.stat().st_mtime_ns==stat.st_mtime_ns
report={'exe':str(exe),'exe_sha256':actual,'exe_bytes':stat.st_size,'exe_stat_unchanged':True,'images_indexed':index,'metadata_selected_bytes_read':readbytes,'seconds':time.monotonic()-t0,'selected':selected,'new_builds':0,'GPU_touches':0,'reuse':'S68 ELF seek parser narrowed to Int8K32Capability; no common archive reads'}
(p/'selection.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['exe_sha256','exe_bytes','images_indexed','metadata_selected_bytes_read','seconds']},indent=2))
