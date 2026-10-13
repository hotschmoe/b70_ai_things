"""NoDri container-only dependency/command preflight; never launches compilation."""
import hashlib,json,re,os,sys
from pathlib import Path

def require(ok,msg):
 if not ok:raise ValueError(msg)
def audit(proof,dependencies,dryrun,resolve):
 current=None;seen=set();dependent=set();records=0;lists={}
 for line in dependencies.splitlines():
  if ': #deps ' in line:
   current=line.split(': #deps ',1)[0];require(current in proof['object_sha256'],'Unexpected dependency object '+current);seen.add(current);lists[current]=[];continue
  if not line.strip():current=None;continue
  require(current is not None and line.startswith('    '),'Unparsed Ninja dependency record')
  name=os.path.normpath(line.strip()); records+=1;lists[current].append(name)
  if name.startswith('/src/'):
   leaf=name[5:];require(leaf in proof['source_tree'],'Foreign source dependency '+name)
   if leaf in proof['changed_files']:
    require(resolve(name)==proof['expected_child_sources'][leaf],'Exact proposed changed source/header required '+name);dependent.add(current);continue
   require(resolve(name)==proof['source_tree'][leaf],'Source/header dependency changed '+name)
  elif name.startswith('/ggml/'):
   leaf=name[6:];require(leaf in proof['ggml_tree'] and resolve(name)==proof['ggml_tree'][leaf],'ggml source/header dependency changed '+name)
  elif name.startswith('/build/'):
   leaf=name[7:];require(leaf in proof['build_dependency_sha256'] and resolve(name)==proof['build_dependency_sha256'][leaf],'Generated dependency changed '+name)
  else:require(name.startswith(('/opt/intel/','/usr/')),'Unadmitted dependency outside identical compiler image '+name)
 require(seen==set(proof['object_sha256']) and dependent==set(proof['changed_objects']),'Every object needs authoritative deps; all six changed-file/header consumers must be rebuilt')
 commands=[line.split('] ',1)[1] for line in dryrun.splitlines() if re.match(r'^\[[0-9]+/[0-9]+\] ',line)]
 compiles=[line for line in commands if ' -c ' in line and ('icpx' in line or 'icx' in line)]
 require(len(compiles)==len(proof['changed_objects']) and all(any(name in command for command in compiles) for name in proof['changed_objects']),'Exactly six authoritative changed consumers must compile')
 require(not any('cmake --regenerate' in line or 'RERUN_CMAKE' in line for line in commands),'Configuration must settle before exact incremental admission')
 archives=[line for line in commands if ' -E rm -f lib' in line or 'ar qc ' in line];require(len(archives)==len(proof['scheduled_archives']) and all(sum(('rm -f '+name) in x for x in archives)==1 for name in proof['scheduled_archives']),'Exactly three changed-consumer archives must rebuild')
 require(len(commands)==len(proof['changed_objects'])+len(proof['scheduled_archives'])+8,'Expected exactly six compiles, three archives and eight fresh ELF links')
 return dict(passed=True,dependency_lists_sha256={obj:hashlib.sha256(json.dumps(names,separators=(',',':')).encode()).hexdigest() for obj,names in lists.items()},dependency_objects=len(seen),dependency_records=records,changed_objects=proof['changed_objects'],scheduled_commands=commands,unchanged_source_headers_rechecked=True,all_changed_header_consumers_accounted=True,compiler_system_dependencies_bound_by_identical_image=True)
def main():
 proof=json.loads(Path('/proof.json').read_bytes())
 final='--final' in sys.argv
 result=audit(proof,Path('/build/incremental-final-dependencies.txt' if final else '/build/incremental-dependencies.txt').read_text(),Path('/build/incremental-dryrun.txt').read_text(),lambda name:hashlib.sha256(Path(name).read_bytes()).hexdigest())
 Path('/build/incremental-final-audit.json' if final else '/build/incremental-audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='ascii')
if __name__=='__main__':main()
