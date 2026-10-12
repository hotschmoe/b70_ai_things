"""NoDri container-only dependency/command preflight; never launches compilation."""
import hashlib,json,re,os
from pathlib import Path

def require(ok,msg):
 if not ok:raise ValueError(msg)
def audit(proof,dependencies,dryrun,resolve):
 current=None;seen=set();dependent=set();records=0
 for line in dependencies.splitlines():
  if ': #deps ' in line:
   current=line.split(': #deps ',1)[0];require(current in proof['object_sha256'],'Unexpected dependency object '+current);seen.add(current);continue
  if not line.strip():current=None;continue
  require(current is not None and line.startswith('    '),'Unparsed Ninja dependency record')
  name=os.path.normpath(line.strip()); records+=1
  if name.startswith('/src/'):
   leaf=name[5:];require(leaf in proof['source_tree'],'Foreign source dependency '+name)
   if leaf in ('include/strata/prefill/host60.hpp','sycl/src/prefill/prefill.cpp'):dependent.add(current);continue
   require(resolve(name)==proof['source_tree'][leaf],'Source/header dependency changed '+name)
  elif name.startswith('/ggml/'):
   leaf=name[6:];require(leaf in proof['ggml_tree'] and resolve(name)==proof['ggml_tree'][leaf],'ggml source/header dependency changed '+name)
  elif name.startswith('/build/'):
   leaf=name[7:];require(leaf in proof['build_dependency_sha256'] and resolve(name)==proof['build_dependency_sha256'][leaf],'Generated dependency changed '+name)
  else:require(name.startswith(('/opt/intel/','/usr/')),'Unadmitted dependency outside identical compiler image '+name)
 require(seen==set(proof['object_sha256']) and dependent==set(proof['changed_objects']),'Every reused object needs authoritative deps; changed body may affect exactly main/Prefill objects')
 commands=[line.split('] ',1)[1] for line in dryrun.splitlines() if re.match(r'^\[[0-9]+/[0-9]+\] ',line)]
 compiles=[line for line in commands if ' -c ' in line and ('icpx' in line or 'icx' in line)]
 require(len(compiles)==2 and all(any(name in command for command in compiles) for name in proof['changed_objects']),'Exactly current main and Prefill compilation required')
 require(not any('cmake --regenerate' in line or 'RERUN_CMAKE' in line for line in commands),'Configuration must settle before exact incremental admission')
 archives=[line for line in commands if ' -E rm -f lib' in line or 'ar qc ' in line];require(len(archives)==1 and 'libstrata_prefill.a' in archives[0],'Only Prefill archive may be rebuilt')
 require(len(commands)==11,'Expected exactly two compiles, one Prefill archive and eight ELF links')
 return dict(passed=True,dependency_objects=len(seen),dependency_records=records,changed_objects=proof['changed_objects'],scheduled_commands=commands,source_headers_unchanged=True,compiler_system_dependencies_bound_by_identical_image=True)
def main():
 proof=json.loads(Path('/proof.json').read_bytes())
 result=audit(proof,Path('/build/incremental-dependencies.txt').read_text(),Path('/build/incremental-dryrun.txt').read_text(),lambda name:hashlib.sha256(Path(name).read_bytes()).hexdigest())
 Path('/build/incremental-audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='ascii')
if __name__=='__main__':main()
