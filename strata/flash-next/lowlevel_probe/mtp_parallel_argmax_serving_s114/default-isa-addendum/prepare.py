from pathlib import Path
import ast,hashlib,json,struct
P=Path(__file__).resolve().parent;R=P.parent;A=R/'mtp-parallel-argmax-main-s114-audit'
sha=lambda b:hashlib.sha256(b).hexdigest()
def cstr(buf,start):
 end=buf.find(b'\0',start);assert end>=start;return buf[start:end]
helper=R/'q6-a4-s104-compiled-audit-prep/audit.py';node=next(n for n in ast.parse(helper.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='sections');exec(compile(ast.Module(body=[node],type_ignores=[]),str(helper)+':sections-only','exec'))
f=json.loads((A/'selection.json').read_text());c=json.loads((A/'comparison.json').read_text());rows={}
expected={'sampler':'72d196b7cf23fef008f09d411c25ddae5f7ec7b77337318f41321b622f9dab8b','probability':'22dcc58a0d0df31f82667b16328123fd8bd6bb8a2e360058bfa1705214646522'}
for arm,expectedsha in expected.items():
 src=A/(arm+'.zebin');image=src.read_bytes();assert sha(image)==expectedsha==f['selected'][arm]['image_sha256'];name=f['selected'][arm]['kernel'];ss,_=sections(image);t=ss['.text.'+name];alias=bytearray(image)
 so=struct.unpack_from('<Q',image,40)[0];se,sn,si=struct.unpack_from('<HHH',image,58);st=struct.unpack_from('<IIQQQQIIQQ',image,so+si*se);at=st[4]+t[0];old=cstr(alias,at);new=('.text.s114_'+arm).encode();assert old==('.text.'+name).encode()and len(new)<len(old)
 alias[at:at+len(old)]=new+b'\0'*(len(old)-len(new));diff=[i for i,(x,y)in enumerate(zip(image,alias))if x!=y];assert all(st[4]<=i<st[4]+st[5]for i in diff);assert alias[t[4]:t[4]+t[5]]==image[t[4]:t[4]+t[5]]
 out=P/(arm+'-alias.zebin');assert not out.exists();out.write_bytes(alias)
 oldimage=(A/('S107-'+arm+'.zebin')).read_bytes();assert sha(oldimage)==f['S107_selected'][arm]['image_sha256']
 rows[arm]={'source_path':str(src),'source_sha256':expectedsha,'reference_path':str(A/('S107-'+arm+'.zebin')),'reference_sha256':sha(oldimage),'kernel':name,'alias_sha256':sha(alias),'alias_text_byteexact':True,'only_section_name_changed':True,'alias_changed_bytes':len(diff),'raw_first_comparison':c[arm]}
(P/'bindings.json').write_text(json.dumps({'source_selection_sha256':sha((A/'selection.json').read_bytes()),'first_comparison_sha256':sha((A/'comparison.json').read_bytes()),'sections_helper_sha256':sha(helper.read_bytes()),'rows':rows},indent=2)+'\n')
print('PREPARED_EXACT_TWO_SELECTED_COPY_ALIASES')
