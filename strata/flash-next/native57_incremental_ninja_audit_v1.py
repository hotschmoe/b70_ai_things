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
   if leaf=='sycl/src/program/generate.cpp':dependent.add(current);continue
   require(resolve(name)==proof['source_tree'][leaf],'Source/header dependency changed '+name)
  elif name.startswith('/ggml/'):
   leaf=name[6:];require(leaf in proof['ggml_tree'] and resolve(name)==proof['ggml_tree'][leaf],'ggml source/header dependency changed '+name)
  elif name.startswith('/build/'):
   leaf=name[7:];require(leaf in proof['build_dependency_sha256'] and resolve(name)==proof['build_dependency_sha256'][leaf],'Generated dependency changed '+name)
  else:require(name.startswith(('/opt/intel/','/usr/')),'Unadmitted dependency outside identical compiler image '+name)
 require(seen==set(proof['object_sha256']) and dependent=={proof['changed_object']},'Every reused object needs authoritative deps; changed body may affect exactly its object')
 commands=[line.split('] ',1)[1] for line in dryrun.splitlines() if re.match(r'^\[[0-9]+/[0-9]+\] ',line)]
 compiles=[line for line in commands if ' -c ' in line and ('icpx' in line or 'icx' in line)]
 require(len(compiles)==1 and '/src/sycl/src/program/generate.cpp' in compiles[0] and proof['changed_object'] in compiles[0],'Ninja must schedule exactly the changed main-program compilation')
 require(not any('cmake --regenerate' in line or 'RERUN_CMAKE' in line for line in commands),'Configuration must settle before exact incremental admission')
 require(not any(' -E rm -f lib' in line or 'ar qc ' in line for line in commands),'All static archives must be reused unchanged')
 require(len(commands)==9,'Expected exactly one main compile and eight ELF links')
 return dict(passed=True,dependency_objects=len(seen),dependency_records=records,changed_object=proof['changed_object'],scheduled_commands=commands,source_headers_unchanged=True,compiler_system_dependencies_bound_by_identical_image=True)
def main():
 proof=json.loads(Path('/proof.json').read_bytes())
 result=audit(proof,Path('/build/incremental-dependencies.txt').read_text(),Path('/build/incremental-dryrun.txt').read_text(),lambda name:hashlib.sha256(Path(name).read_bytes()).hexdigest())
 Path('/build/incremental-audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='ascii')
if __name__=='__main__':main()
