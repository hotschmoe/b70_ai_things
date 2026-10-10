"""Bounded read-only extraction of original executed ELF symbol-sized images."""
import hashlib,json,re,struct
from pathlib import Path
from serial37_canonical_json_v3 import read_unique,canonical
import qualify_owned_indexer_half_control_v2 as q
from half37_spirv_text_recipe_v3 import MODULES
ROOT=Path('/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/owned-indexer-half-control-v2-run-v1')
REPORT_SHA='5e66575dcf49c66fd1118c9af14e118217fa672ea5ed42217e506cba1b110cc3'
HELPER_SHA='4d86d9b8d50655949cd30ded27e20337e7ed03ab20549ba29915dae9abcedb63'
PUBLIC=ROOT.parent/'owned-indexer-half-control-v2-readonly-binding-v1.json'
PUBLIC_SHA='e1789d5c870a522cbb23931a96b3e87c470f5dffc16046f93e711b134c9cb727'
SECTION='__CLANG_OFFLOAD_BUNDLE__sycl-spir64'
MAX_ELF=8<<20
require=q.require

def digest(raw):return hashlib.sha256(raw).hexdigest()
def consume(path,limit):
 path=Path(path);require(path.is_file() and not path.is_symlink() and path.absolute()==path.resolve(),'Original regular nonalias code artifact required');before=path.stat();require(before.st_size<=limit,'Bounded actual code artifact before read');raw=path.read_bytes();require(path.stat()==before and len(raw)==before.st_size and path.read_bytes()==raw and path.stat()==before,'Consumed original artifact changed');return raw

def span(raw,offset,size):
 require(type(offset)is int and type(size)is int and offset>=0 and size>=0 and offset<=len(raw) and size<=len(raw)-offset,'Exact file-backed bounded ELF extent required');return raw[offset:offset+size]
def string(raw,offset):
 require(0<=offset<len(raw),'Bounded symbol/string index');end=raw.find(b'\0',offset);require(end>=offset,'Terminated original ELF string');return raw[offset:end].decode('ascii')
def spirv(raw):
 require(len(raw)>=20 and len(raw)%4==0 and len(raw)<=1<<20,'Bounded complete SPIRV words required');words=struct.unpack('<'+'I'*(len(raw)//4),raw);require(words[0]==0x07230203 and words[1]>>16==1 and words[3]>0 and words[4]==0,'Actual SPIRV header/magic required');entries=[];instructions=[];i=5
 while i<len(words):
  count,opcode=words[i]>>16,words[i]&65535;require(count>0 and count<=len(words)-i,'Bounded nonzero SPIRV instruction width');operands=list(words[i+1:i+count]);instructions.append({'word_offset':i,'opcode':opcode,'operands':operands})
  if opcode==15:
   require(count>=4,'Complete original entrypoint instruction');data=struct.pack('<'+'I'*len(operands[2:]),*operands[2:]);entries.append(string(data,0))
  i+=count
 require(i==len(words) and entries and len(entries)==len(set(entries)),'Exact complete unique entrypoint roster required')
 return {'entrypoints':entries,'instructions':instructions,'word_count':len(words),'version_word':words[1],'generator_word':words[2],'bound':words[3]}

def elf_images(raw):
 require(type(raw)is bytes and 64<=len(raw)<=MAX_ELF and raw[:7]==b'\x7fELF\x02\x01\x01','Original bounded ELF64LE required');require(struct.unpack_from('<H',raw,18)[0]==62,'Actual x86_64 host executable required');offset=struct.unpack_from('<Q',raw,40)[0];size,count,names=struct.unpack_from('<HHH',raw,58);require(size==64 and 1<count<=256 and 0<names<count,'Bounded original section table required');span(raw,offset,size*count);headers=[struct.unpack_from('<IIQQQQIIQQ',raw,offset+i*size)for i in range(count)];strings=span(raw,headers[names][4],headers[names][5]);sections=[string(strings,h[0])for h in headers];require(sections.count(SECTION)==1,'Exactly one actual offload section required');image_section=sections.index(SECTION);images={}
 for h in headers:
  if h[1]!=2:continue
  require(h[9]==24 and h[5]%24==0 and h[5]//24<=8192 and h[6]<count,'Bounded exact original symbol table required');symbols=span(raw,h[4],h[5]);s=headers[h[6]];names_data=span(raw,s[4],s[5])
  for pos in range(0,len(symbols),24):
   name,info,other,index,value,extent=struct.unpack_from('<IBBHQQ',symbols,pos);name=string(names_data,name);match=re.fullmatch(r'\.sycl_offloading\.([0-9]+)\.data',name)
   if not match:continue
   number=int(match[1]);require(number not in images and index==image_section and info&15==1 and extent>0,'Exact unique original embedded object symbols required');section=headers[index];require(value>=section[3] and value-section[3]+extent<=section[5],'Original symbol extent outside offload section');file_offset=section[4]+value-section[3];payload=span(raw,file_offset,extent);decoded=spirv(payload);images[number]={'symbol':name,'section':SECTION,'offset':file_offset,'bytes':extent,'sha256':digest(payload),'entrypoints':decoded['entrypoints'],'SPIRV':decoded,'raw':payload}
 require(images,'Original embedded SPIRV object roster missing');intervals=sorted((row['offset'],row['offset']+row['bytes'])for row in images.values());require(all(a[1]<=b[0]for a,b in zip(intervals,intervals[1:])),'Overlapping original embedded image extents')
 selected={index:images[index]for index in MODULES if index in images};require(set(selected)==set(MODULES),'Complete original six module roster required')
 for index,symbol in MODULES.items():require(selected[index]['entrypoints']==[symbol],'Original embedded image entrypoint substitution')
 return selected

def original_binding(root=ROOT,*,requalify=True):
 root=Path(root).resolve();require(root==ROOT and digest(consume(root/'report.json',4<<20))==REPORT_SHA and digest(consume(PUBLIC,4<<20))==PUBLIC_SHA,'Exact original closed report/public binding required');report=read_unique(root/'report.json');saved=read_unique(PUBLIC);require(report['passed'] is True and report['errors']==[] and report['helper_sha256']==HELPER_SHA and saved['report_sha256']==REPORT_SHA and saved['actual_own_half_conversion_observed'] is True,'Actual original successful helper proof required');require(report['compile_argv']==q.leaf_argv() and report['source_binding']==q.source_binding(),'Original actual source/object/link flags differ')
 if requalify:
  current=q.finalized_binding(root);expected={k:v for k,v in saved.items()if k not in ('started_epoch','finished_epoch')};require(canonical(current)==canonical(expected),'Current public original runtime/source proof differs')
 helper=root/'build/owned-indexer-half37';raw=consume(helper,MAX_ELF);require(digest(raw)==HELPER_SHA,'Actual original executed helper substituted');return {'root':str(root),'report_sha256':REPORT_SHA,'public_sha256':PUBLIC_SHA,'helper_sha256':HELPER_SHA,'actual_compile_argv':report['compile_argv'],'actual_runtime_library_binding':report['runtime_binding_after'],'current_full_runtime_requalification_performed':requalify},raw

def extract(output):
 binding,raw=original_binding();images=elf_images(raw);out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False);record={}
 for index,row in images.items():
  name='module-'+str(index)+'.spv'
  with (out/name).open('xb')as f:f.write(row['raw'])
  record[str(index)]={k:v for k,v in row.items()if k not in ('raw','SPIRV')};record[str(index)]['file']=name
 report={'schema':4,'original_binding':binding,'images':record,'original_embedded_images_extracted':True,'original_runtime_JIT_ISA_observed':False,'conversion_removed_before_embedded_SPIRV_claimed':False,'full_model_math_qualified':False};(out/'extraction.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');require(consume(ROOT/'build/owned-indexer-half37',MAX_ELF)==raw,'Original executed helper changed during extraction');return report

def finalized_binding(output):
 out=Path(output).resolve();report_raw=consume(out/'extraction.json',4<<20);report=read_unique(out/'extraction.json');require(consume(out/'extraction.json',4<<20)==report_raw,'Extraction record changed while consumed');require(type(report.get('schema'))is int and report['schema']==4,'Exact extraction schema required');binding,raw=original_binding();require(canonical(binding)==canonical(report['original_binding']),'Exact current original closed source/runtime binding required');images=elf_images(raw);require(set(report['images'])=={str(i)for i in MODULES},'Exact reported six image roster required');require({p.name for p in out.iterdir()}=={'extraction.json',*('module-'+str(i)+'.spv'for i in MODULES)},'Exact extracted original file roster required')
 for index,row in images.items():
  saved=report['images'][str(index)];expected={k:v for k,v in row.items()if k not in ('raw','SPIRV')};expected['file']='module-'+str(index)+'.spv';require(canonical(saved)==canonical(expected) and consume(out/expected['file'],1<<20)==row['raw'],'Exact original embedded extents/bytes/hash/entrypoint substituted')
 require(report['original_embedded_images_extracted'] is True and report['original_runtime_JIT_ISA_observed'] is False and report['conversion_removed_before_embedded_SPIRV_claimed'] is False and report['full_model_math_qualified'] is False,'Original embedded code cannot grant JIT/model authority');require(consume(ROOT/'build/owned-indexer-half37',MAX_ELF)==raw,'Original ELF changed during admission');require(consume(out/'extraction.json',4<<20)==report_raw,'Extraction record changed during admission');return {'extraction_sha256':digest(report_raw),'original_helper_sha256':HELPER_SHA,'original_embedded_images_extracted':True,'original_runtime_JIT_ISA_observed':False,'full_model_math_qualified':False}
