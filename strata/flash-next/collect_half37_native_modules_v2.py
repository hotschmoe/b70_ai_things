"""Strict source/module/kernel/submission byte join; outer actual lifecycle separate."""
import hashlib,json
from pathlib import Path
from serial37_canonical_json_v3 import read_unique
from half37_original_elf_spirv_v4 import MODULES,HELPER_SHA,ROOT,MAX_ELF,consume,elf_images
ROLES={'expression':(21,33),'store':(24,36),'load':(25,38)}
def require(ok,message):
 if not ok:raise ValueError(message)
def digest(raw):return hashlib.sha256(raw).hexdigest()
def records(root):
 p=Path(root)/'events.jsonl';raw=consume(p,4<<20);require(raw.endswith(b'\n'),'Actual terminal trace newline missing');rows=[]
 from serial37_canonical_json_v3 import unique_object,finite_constant
 for line in raw.splitlines():rows.append(json.loads(line,object_pairs_hook=unique_object,parse_constant=finite_constant))
 require(rows and len(rows)<=8193 and [r['sequence']for r in rows]==list(range(1,len(rows)+1)) and all(type(r['sequence'])is int and type(r['pid'])is int and r['pid']>0 for r in rows) and len({r['pid']for r in rows})==1,'Exact bounded ordered process-local trace required');require(rows[-1]['event']=='terminal' and rows[-1]['failed']is False and not any(r['event']in ('error','terminal')for r in rows[:-1]),'Capture failure or partial process trace');return rows

def structural_binding(root,*,synthetic=False):
 root=Path(root).resolve();rows=records(root);modules={};kernels={};submitted={};files={'events.jsonl'}
 for r in rows[:-1]:
  kind=r['event']
  if kind=='symbol_lookup':require(type(r['name'])is str and r['name'].startswith('ze') and type(r['library'])is str and r['library'].startswith('/'),'Actual specific-loader function resolution absent')
  elif kind=='proc_table':require(type(r['version'])is int and type(r['name'])is str,'Typed actual dispatch table source required')
  elif kind=='module':
   require(type(r['id'])is int and r['id']>0 and r['id']not in modules and r['context'] and r['device'],'Unique current module generation/context/device required');require(type(r['device_properties_rc'])is int and r['device_properties_rc']==0,'Actual device source property query failed');modules[r['id']]=r
   for label in ('input','native'):
    name=r[label];require(type(name)is str and Path(name).name==name and name and name not in files,'Exact unique actual native/input dump basename');data=consume(root/name,16<<20);require(data,'Complete binary dump required');files.add(name)
   require(consume(root/r['native'],16<<20)[:4]==b'\x7fELF','Actual native module must be ELF')
  elif kind=='kernel':require(type(r['id'])is int and r['id']not in kernels and r['module']in modules and type(r['name'])is str,'Actual kernel/current module owner missing');kernels[r['id']]=r
  elif kind=='append':require(r['kernel']in kernels and type(r['list'])is int and r['list']>0,'Original append owner missing')
  elif kind=='submitted':
   require(r['kernel']in kernels and r['module']==kernels[r['kernel']]['module'] and r['name']==kernels[r['kernel']]['name'] and r['via']in ('queue','immediate','immediate-graph') and type(r['list'])is int and r['list']>0,'Actual submitted kernel/module/name association differs');submitted[r['name']]=submitted.get(r['name'],0)+1
  else:raise ValueError('Unexpected interposer trace record '+str(kind))
 require({p.name for p in root.iterdir()}==files and all(p.is_file()and not p.is_symlink()for p in root.iterdir()),'Exact complete regular native capture file tree required');require(type(rows[-1]['events'])is int and rows[-1]['events']==len(rows)-1 and type(rows[-1]['binary_bytes'])is int and rows[-1]['binary_bytes']==sum((root/name).stat().st_size for name in files if name!='events.jsonl') and rows[-1]['binary_bytes']<=64<<20,'Actual event/binary bound/accounting differs')
 result={'PID':rows[0]['pid'],'modules':modules,'kernels':kernels,'submitted':submitted,'trace_sha256':digest(consume(root/'events.jsonl',4<<20)),'synthetic_CPU_ABI_fixture_only':synthetic,'full_model_math_or_JIT_cause_qualified':False,'actual_GPU_execution_proven_by_this_structural_reader':False}
 return result

def collect(root,runtime_log):
 before=structural_binding(root);raw=consume(ROOT/'build/owned-indexer-half37',MAX_ELF);require(digest(raw)==HELPER_SHA,'Unchanged original executed helper required');images=elf_images(raw);wanted={name:i for i,name in MODULES.items()};roles={};output={};log=Path(runtime_log).read_text()
 require('HALF37_INTERPOSER'not in log and 'HALF37_ERROR'not in log,'Observer/runtime error marker present');require('HALF37_RESULT frames=3 graph_retired=1 owned_allocations_freed=1'in log,'Actual unchanged helper complete three frames required')
 for name,count in before['submitted'].items():
  if name not in wanted:continue
  index=wanted[name];objects=[k for k in before['kernels'].values()if k['name']==name];require(objects,'Actual submitted role kernel missing')
  for k in objects:
   m=before['modules'][k['module']];require(m['device_id']==0xe223 and type(m['device_id'])is int and m['vendor_id']==0x8086 and type(m['vendor_id'])is int and m['format']==0 and m['constants_present']is False and m['pnext_present']is False,'Actual original code/device/constant source association incomplete');require(consume(Path(root)/m['input'],16<<20)==images[index]['raw'],'Actual JIT input differs original ELF image');native=consume(Path(root)/m['native'],16<<20);output[name]={'module_id':m['id'],'input_sha256':digest(images[index]['raw']),'native_sha256':digest(native),'native_bytes':len(native),'context':m['context'],'device':m['device'],'device_uuid':m['device_uuid'],'build_flags':m['build_flags'],'submitted':count}
 for role,indices in ROLES.items():
  names=[MODULES[i]for i in indices];count=sum(before['submitted'].get(name,0)for name in names);require(count>=3,'Missing actual direct plus two-replay role submission '+role);roles[role]=count
 require(consume(ROOT/'build/owned-indexer-half37',MAX_ELF)==raw and structural_binding(root)==before,'Original ELF/native capture changed during read');return {'actual_native_module_bytes':output,'actual_role_submissions':roles,'current_original_ELF_inputs_joined':True,'actual_kernel_execution_proven_by_outer_lifecycle_and_raw_equality_required':True,'original_historical_run_JIT_code_observed':False,'normal_model_graph_or_full_math_qualified':False,'JIT_conversion_cause_qualified':False}
