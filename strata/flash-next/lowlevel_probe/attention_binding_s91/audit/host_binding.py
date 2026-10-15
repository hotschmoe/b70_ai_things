from pathlib import Path
import struct,json,hashlib
P=Path(__file__).resolve().parent;B=P.parent/'build';out={}
for fname in ['candidate.o','public-current-member.o']:
 data=(B/fname).read_bytes();so=struct.unpack_from('<Q',data,40)[0];se,sn,si=struct.unpack_from('<HHH',data,58);ss=[struct.unpack_from('<IIQQQQIIQQ',data,so+k*se)for k in range(sn)];ns=ss[si];strings=data[ns[4]:ns[4]+ns[5]];names=[strings[x[0]:].split(b'\0')[0].decode()for x in ss];sy=ss[names.index('.symtab')];st=ss[sy[6]];strings=data[st[4]:st[4]+st[5]];symbols=[]
 for off in range(sy[4],sy[4]+sy[5],sy[9]):
  z=struct.unpack_from('<IBBHQQ',data,off);symbols.append({'name':strings[z[0]:].split(b'\0')[0].decode(),'bind':z[1]>>4,'type':z[1]&15,'section':z[3],'offset':z[4],'size':z[5]})
 targets=[z for z in symbols if'CompileTimeKernelInfoI16dpct_kernel_name'in z['name']and'attn_chunk_kernel_2ad0a4'in z['name']];assert len(targets)==1,targets;t=targets[0];rel=[]
 for idx,section in enumerate(ss):
  if section[1]!=4 or section[7]!=t['section']:continue
  for off in range(section[4],section[4]+section[5],section[9]):
   ro,ri,add=struct.unpack_from('<QQq',data,off)
   if not(t['offset']<=ro<t['offset']+t['size']):continue
   target=symbols[ri>>32];entry={'offset_in_info':ro-t['offset'],'reloc_type':ri&0xffffffff,'target_symbol':target['name'],'target_section':names[target['section']]if target['section']<len(names)else target['section'],'addend':add}
   if target['section']<len(ss):
    sec=ss[target['section']];pos=sec[4]+target['offset']+add
    if 0<=pos<len(data):entry['cstring_candidate']=data[pos:pos+1000].split(b'\0')[0].decode(errors='backslashreplace')
   rel.append(entry)
 out[fname]={'target_weak_compiletime_symbol':t,'relocations':rel,'related_shared_info_symbols':[z for z in symbols if'getDeviceKernelInfo'in z['name']and'attn_chunk_kernel_2ad0a4'in z['name']]}
out['same_compiletime_weak_symbol']=out['candidate.o']['target_weak_compiletime_symbol']['name']==out['public-current-member.o']['target_weak_compiletime_symbol']['name']
(P/'host-binding.json').write_text(json.dumps(out,indent=2,ensure_ascii=True)+'\n',encoding='ascii');print(json.dumps(out,indent=2,ensure_ascii=True))
