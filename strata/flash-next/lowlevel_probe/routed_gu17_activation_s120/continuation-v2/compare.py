from pathlib import Path
import hashlib,struct
sha=lambda b:hashlib.sha256(b).hexdigest()
def cstring(data,start):
 end=data.find(b'\0',start);assert end>=start;return data[start:end]
def sections(image):
 so=struct.unpack_from('<Q',image,40)[0];se,sn,si=struct.unpack_from('<HHH',image,58);assert se==64
 rows=[struct.unpack_from('<IIQQQQIIQQ',image,so+k*se)for k in range(sn)];z=rows[si];ns=image[z[4]:z[4]+z[5]]
 names=[cstring(ns,z[0]).decode()for z in rows]
 return rows,names,{n:(z,image[z[4]:z[4]+z[5]])for n,z in zip(names,rows)}
def symbols(image):
 rows,names,ss=sections(image);z=ss['.symtab'][0];st=rows[z[6]];ns=image[st[4]:st[4]+st[5]];out=[]
 for at in range(z[4],z[4]+z[5],z[9]):
  name,info,other,section,value,size=struct.unpack_from('<IBBHQQ',image,at)
  out.append((cstring(ns,name).decode(),info,other,names[section],value,size))
 return out
def raw_inventory(reference,current,new_kernel):
 old=reference.read_bytes();new=current.read_bytes();_,onames,os=sections(old);_,nnames,ns=sections(new)
 old_kernel=next(n[len('.text.'):]for n in onames if n.startswith('.text.')and n!='.text.Intel_Symbol_Table_Void_Program')
 a={n.replace(old_kernel,'TARGET_KERNEL'):(n,z,b)for n,(z,b)in os.items()};b={n.replace(new_kernel,'TARGET_KERNEL'):(n,z,x)for n,(z,x)in ns.items()};rows=[]
 for key in sorted(set(a)|set(b)):
  row={'normalized_section':key}
  for label,table in [('reference',a),('current',b)]:
   if key in table:
    name,header,data=table[key];row[label]={'name':name,'header':header,'sha256':sha(data),'bytes':len(data)}
  if key in a and key in b:
   x=a[key][2];y=b[key][2];row['raw_exact']=x==y
   if key.startswith('.text.'):
    row['all_changed_byte_offsets_common_length']=[i for i,(u,v)in enumerate(zip(x,y))if u!=v]
    row['length_delta']=len(y)-len(x)
  rows.append(row)
 return {'reference_path':str(reference),'current_path':str(current),'reference_sha256':sha(old),'current_sha256':sha(new),'ELF_headers_raw_exact':old[:64]==new[:64],'reference_ELF_header_hex':old[:64].hex(),'current_ELF_header_hex':new[:64].hex(),'sections':rows}
def compare_images(reference,current,expected,new_kernel,prefix):
 old=reference.read_bytes();new=current.read_bytes();assert sha(old)==expected
 assert old[:40]==new[:40]and old[48:64]==new[48:64],'ELF header difference beyond section-table offset'
 orows,onames,os=sections(old);nrows,nnames,ns=sections(new)
 old_kernel=next(n[len('.text.'):]for n in onames if n.startswith('.text.')and n!='.text.Intel_Symbol_Table_Void_Program')
 norm_old=lambda n:n.replace(old_kernel,'TARGET_KERNEL');norm_new=lambda n:n.replace(new_kernel,'TARGET_KERNEL')
 assert [norm_old(n)for n in onames]==[norm_new(n)for n in nnames],'Section order/names differ beyond target name'
 oldmap={norm_old(n):(z,b)for n,(z,b)in os.items()};newmap={norm_new(n):(z,b)for n,(z,b)in ns.items()};assert oldmap.keys()==newmap.keys(),'Section names differ beyond target name'
 section_rows=[]
 for name,(a,x)in oldmap.items():
  b,y=newmap[name];row={'normalized_section':name,'reference_bytes':len(x),'current_bytes':len(y),'reference_sha256':sha(x),'current_sha256':sha(y),'raw_exact':x==y}
  assert all(a[k]==b[k]for k in [1,2,3,6,7,8,9]),(name,'section attributes')
  if name in ['.text.TARGET_KERNEL','.text.Intel_Symbol_Table_Void_Program']:
   at=prefix if name=='.text.TARGET_KERNEL' else 80
   allowed=[q for word in [at+12,at+28]for q in range(word,word+4)]
   assert len(x)==len(y),(name,'changed instruction section size')
   changed=[i for i,(u,v)in enumerate(zip(x,y))if u!=v];row.update({'changed_byte_offsets':changed,'exact_prefix_through_EOT_and_alignment_bytes':at,'only_allowed_difference':'two previously decoded post-EOT MOV-to-null immediate fields'})
   assert x[:at]==y[:at]and all(i in allowed for i in changed),(name,'computational/tail difference: HOLD target inspection, no automatic decoder/rebuild')
  elif name=='.ze_info':
   assert norm_old(x.decode())==norm_new(y.decode()),'Metadata differs beyond registered name';row['normalized_exact']=True
  elif name=='.symtab':
   normal=lambda values,norm:[(norm(z[0]),z[1],z[2],norm(z[3]),z[4],z[5])for z in values]
   assert normal(symbols(old),norm_old)==normal(symbols(new),norm_new),'Symbol semantic mismatch';row['normalized_exact']=True
  elif name=='.strtab':
   xs=[norm_old(t.decode())for t in x.rstrip(b'\0').split(b'\0')];ys=[norm_new(t.decode())for t in y.rstrip(b'\0').split(b'\0')]
   assert xs==ys,'Strings differ beyond target name';row['normalized_exact']=True
  elif name=='.spv':row['classification']='Embedded SPIR-V raw difference recorded; not emitted AOT instruction bytes; no universal semantic-equivalence claim'
  else:assert x==y,(name,'unclassified actual section difference')
  section_rows.append(row)
 return {'reference':str(reference),'reference_sha256':expected,'raw_full_image_exact':old==new,'raw_target_text_exact':os['.text.'+old_kernel][1]==ns['.text.'+new_kernel][1],'computational_prefix_through_EOT_and_alignment_exact':True,'normalizations':'ONLY registered target name in ze_info/string/symbol/section names; explicit post-EOT immediate fields; embedded SPIR-V raw difference separately recorded','section_comparisons':section_rows}
