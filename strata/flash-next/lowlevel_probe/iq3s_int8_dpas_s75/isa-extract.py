#!/usr/bin/env python3
# Narrow S68 ELF seek extraction; only S75 candidate +control/public identities.
from pathlib import Path
import hashlib,json,struct,time,re,collections
p=Path(__file__).resolve().parent;r=p.parent;b=r/'iq3s-int8-dpas-s75-prep-v2/build';exe=b/'iq3s_dpas_leaf_s75'
sha=lambda b:hashlib.sha256(b).hexdigest()
expected='18045cced6ce5539851a8e9db1e52a40bff6660a5e3aa9af4f0d019dcdae032f'
records={}
for l in(b/'sha256.txt').read_text().splitlines():h,n=l.split(maxsplit=1);records[n]=h
assert records['iq3s_dpas_leaf_s75']==expected
assert records['/src/private_iq3s_dpas_c6.hpp']=='8a490cc2b46b19e5d0a6bd37141f615e267852067126364e374d4bd89e8ab863'
assert records['/src/iq3s_dpas_leaf.cpp']=='e26056959edc34e95928efd11157d0a53585bdc115c2992b6df9bc8eee470d94'
for n in ['/src/private_iq3s_dpas_c6.hpp','/src/iq3s_dpas_leaf.cpp']:
 assert sha((b.parent/Path(n).name).read_bytes())==records[n]
st=exe.stat();readbytes=0;t0=time.monotonic();entries=[]
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
 top,_=elf(0);bundle=top['__CLANG_OFFLOAD_BUNDLE__sycl-spir64_gen'];off=bundle[4];end=off+bundle[5];index=0;selected={}
 while off<end:
  ss,size=elf(off);z=ss.get('.ze_info');info=rd(off+z[4],z[5]).decode()if z else ''
  kernel_info=info.split('kernels_misc_info:')[0]
  names=re.findall(r'^  - name:\s+(\S+)',kernel_info,re.M)
  entries.extend({'name':n,'image':index}for n in names)
  arm=None
  if 'DpasC6' in info:arm='candidate'
  elif 'PrivateC6ILb0EE' in info:arm='original'
  elif 'launch_mmvq_multi_ncILi21ELb0EE' in info and 'dpct_kernel_scalarILi21EESK_ILi6EESK_ILi0EESK_ILi1EEEE' in info:arm='public'
  if arm:
   assert arm not in selected,(arm,index)
   data=rd(off,size)
   needle='DpasC6'if arm=='candidate'else'PrivateC6ILb0EE'if arm=='original'else'mmvq_multi_kernel_3404b2'
   kernel_sections={n:s for n,s in ss.items()if n.startswith('.text.')and needle in n};assert len(kernel_sections)==1
   name,s=next(iter(kernel_sections.items()));text=data[s[4]:s[4]+s[5]]
   op=ss.get('.misc.buildOptions');options=rd(off+op[4],op[5]).decode().strip('\0')if op else None
   (p/(arm+'.zebin')).write_bytes(data);(p/(arm+'.ze_info')).write_text(info)
   selected[arm]={'index':index,'offset':off,'zebin_bytes':size,'zebin_sha256':sha(data),'kernel':name.removeprefix('.text.'),'text_section':name,'text_bytes':len(text),'text_sha256':sha(text),'build_options':options,'resource_metadata':info}
  off=(off+size+15)//16*16;index+=1
 assert set(selected)=={'public','original','candidate'}
 assert selected['public']['zebin_sha256']=='bad2ccded9f4136e6809825844815652eba954f059b807774c45d7916d92beb4'
assert exe.stat().st_size==st.st_size and exe.stat().st_mtime_ns==st.st_mtime_ns
counts=collections.Counter(e['name']for e in entries)
duplicates={n:c for n,c in counts.items()if c>1}
assert counts[selected['candidate']['kernel']]==1 and counts[selected['original']['kernel']]==1 and counts[selected['public']['kernel']]==1
report={'exe':str(exe),'exe_sha256_from_actual_build_record':expected,'exe_bytes':st.st_size,'exe_mtime_ns':st.st_mtime_ns,'exe_stat_unchanged':True,'full_exe_or_common_archive_hash_not_repeated':True,'build_records_sha256':sha((b/'sha256.txt').read_bytes()),'source_pins':{n:h for n,h in records.items()if n.startswith('/src/')},'images_indexed':index,'metadata_selected_bytes_read':readbytes,'seconds':time.monotonic()-t0,'selected':selected,'public_exact_S65_image_match':True,'kernel_name_duplicate_counts':duplicates,'entries':entries,'new_builds':0,'GPU_touches':0}
(p/'selection.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'images':index,'seconds':report['seconds'],'public_exact_S65_image_match':True,'selected':{a:{k:v[k]for k in ['index','kernel','zebin_bytes','text_bytes','zebin_sha256','text_sha256','resource_metadata']}for a,v in selected.items()}},indent=2))
