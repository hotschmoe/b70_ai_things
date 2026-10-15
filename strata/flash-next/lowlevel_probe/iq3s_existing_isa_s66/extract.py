from pathlib import Path
import struct,json,hashlib
P=Path(__file__).resolve().parent
E=P.parent/'iq4xs-rpw2-s59-prep/build/iq4xs_rpw_leaf_s59'
expected='56e551afe6faf58d377340dd8a248674ef98e7cebb2e4f07011629ee6bfc4814'
readbytes=0
with E.open('rb') as f:
 def rd(off,n):
  global readbytes
  f.seek(off); b=f.read(n); assert len(b)==n; readbytes+=n; return b
 def elf(base):
  h=rd(base,64); assert h[:6]==b'\x7fELF\x02\x01'
  so=struct.unpack_from('<Q',h,40)[0]; se,sn,si=struct.unpack_from('<HHH',h,58)
  ss=rd(base+so,se*sn); sh=[struct.unpack_from('<IIQQQQIIQQ',ss,i*se) for i in range(sn)]
  s=sh[si]; names=rd(base+s[4],s[5])
  sections={names[s[0]:].split(b'\0')[0].decode():s for s in sh}
  end=max(so+se*sn,max(s[4]+s[5] for s in sh if s[1]!=8))
  return sections,end
 top,_=elf(0); sect=top['__CLANG_OFFLOAD_BUNDLE__sycl-spir64_gen']; off,end=sect[4],sect[4]+sect[5]
 selected=[]; index=0
 while off<end:
  sh,size=elf(off)
  z=sh.get('.ze_info'); info=rd(off+z[4],z[5]).decode() if z else ''
  if 'launch_mmvq_multi_ncILi21ELb0EE' in info:
   selected.append({'index':index,'offset':off,'size':size,'info':info,'sections':sh})
  off=(off+size+15)//16*16;index+=1
 assert off>=end
 print('images',index,'matching_type21_multi',len(selected))
 for s in selected:
  print(s['index'],s['offset'],s['size'],s['info'][:500])
 (P/'candidates.json').write_text(json.dumps({'exe':str(E),'expected_sha256':expected,'source_identity':'root frozen executable; existing closure SHA, unchanged stat only on reindex','stat':dict(size=E.stat().st_size,mtime_ns=E.stat().st_mtime_ns),'images':index,'metadata_bytes_read':readbytes,'candidates':selected},indent=2)+'\n')
