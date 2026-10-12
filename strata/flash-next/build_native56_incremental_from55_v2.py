#!/usr/bin/env python3
"""Opt-in direct55->56 isolated NoDri object reuse; no all-source fresh-compile claim."""
import argparse, hashlib, json, os, shlex, shutil, subprocess, sys, tempfile, time
from pathlib import Path
import build_native_hc_engine as base
ROOT=base.REPO; HERE=Path(__file__).resolve().parent
PARENT_PLAN=HERE/'native55-hc-prefill-coalescing-engine-build-proposal-v1.json'
CHILD_PLAN=HERE/'native56-hc-prefill-tile4-engine-build-proposal-v1.json'
PARENT_PLAN_SHA='24137c6d3f32721167e71d27c1267dd48f0891e539bceafd496e375614947687'
CHILD_PLAN_SHA='6ddb4574f4ae3288d9e73cbe82bc21434d95f25f2a46a56bbb80b24060100fe8'
CHANGED='sycl/src/kernels/hc_native_composition.cpp'

def require(ok,msg):
 if not ok: raise ValueError(msg)
def sha(path):
 path=Path(path);require(path.is_file() and not path.is_symlink(),'Exact regular file required '+str(path));before=path.stat();value=base.sha(path);after=path.stat();require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),'File changed during hash '+str(path));return value

def tree(root, skip_git=False):
 out={}
 for path in sorted(root.rglob('*')):
  if skip_git and '.git' in path.relative_to(root).parts: continue
  require(not path.is_symlink(),'Symlink not admitted in copied inputs '+str(path))
  if path.is_file(): out[str(path.relative_to(root))]=sha(path)
 return out

def inspect_parent(parent,receipt_sha):
 require(parent.is_absolute() and parent.resolve()==parent,'Canonical current parent required'); require(sha(parent/'receipt.json')==receipt_sha,'Actual parent receipt SHA differs')
 receipt=json.loads((parent/'receipt.json').read_bytes());a=json.loads(PARENT_PLAN.read_bytes());b=json.loads(CHILD_PLAN.read_bytes())
 require(sha(PARENT_PLAN)==PARENT_PLAN_SHA and sha(CHILD_PLAN)==CHILD_PLAN_SHA and sha(parent/'plan.snapshot.json')==PARENT_PLAN_SHA,'Exact direct parent/child plans required')
 for k in ('source_revision','ggml','cmake_flags','build_targets','source_file_sha256','overlay_files'):require(a[k]==b[k],'Unchanged source/build ABI prerequisite differs '+k)
 require(b['patches'][:-1]==a['patches'] and len(b['patches'])==len(a['patches'])+1,'Only direct56 composition delta permitted')
 require([k for k in a['expected_patched_source_sha256'] if a['expected_patched_source_sha256'][k]!=b['expected_patched_source_sha256'][k]]==[CHANGED],'Exactly one composition source differs')
 require(receipt['image']==base.IMAGE and receipt['build_rc']==0 and receipt['devices_exposed'] is False and receipt['external_source_unchanged'] is True and receipt['plan_snapshot_unchanged'] is True and receipt['plan_sha256']==PARENT_PLAN_SHA,'Authoritative same-image completed55 receipt required')
 require(receipt['source_copy']==str(parent/'source') and receipt['ggml_source']==str(parent/'ggml-source') and receipt['build']==str(parent/'build'),'Parent exact isolated paths required')
 require(len(receipt['binary_sha256'])==8 and set(Path(x).name for x in receipt['binary_sha256'])==set(b['build_targets']),'Actual all8 parent executables required')
 for name,want in receipt['binary_sha256'].items():require(Path(name).parent==parent/'build' and sha(name)==want,'Parent executable changed '+name)
 for name,want in a['expected_patched_source_sha256'].items():require(sha(parent/'source'/name)==want,'Parent55 source changed '+name)
 for patch in b['patches']:require(sha(ROOT/patch['path'])==patch['sha256'],'Reviewed current patch changed '+patch['path'])
 build=parent/'build'; require(sha(build/'CMakeCache.txt')==receipt['cmake_cache_sha256'],'Parent cache differs')
 changed_object='CMakeFiles/strata_kernels.dir/src/kernels/hc_native_composition.cpp.o'
 ninja=(build/'build.ninja').read_text(encoding='ascii')
 require(('build '+changed_object+': CXX_COMPILER__strata_kernels_unscanned_Release /src/'+CHANGED+' ||') in ninja,'Authoritative changed object/source Ninja edge required')
 objects={str(p.relative_to(build)):sha(p) for p in sorted(build.rglob('*.o'))};require(len(objects)==124 and changed_object in objects,'Exact current55 object census required')
 archives={str(p.relative_to(build)):sha(p) for p in sorted(build.rglob('*.a'))}; require(archives and (build/'.ninja_deps').is_file() and (build/'.ninja_log').is_file(),'Existing actual Ninja dependencies/log/archives required')
 metadata={name:sha(build/name) for name in ('CMakeCache.txt','build.ninja','.ninja_deps','.ninja_log','compile_commands.json','compiler-version.txt','packages.txt')}
 return a,b,dict(receipt_sha256=receipt_sha,parent_root=str(parent),source_tree=tree(parent/'source',True),ggml_tree=tree(parent/'ggml-source',True),object_sha256=objects,archive_sha256=archives,build_metadata_sha256=metadata,changed_object=changed_object,build_dependency_sha256={str(p.relative_to(build)):sha(p) for p in build.rglob('*') if p.is_file() and not p.is_symlink() and p.suffix not in ('.o','.a') and p.name not in a['build_targets']},compiler_image=base.IMAGE)

def source_closure():
 path=HERE/'source56-incremental-from55-build-recipe-v2.json'
 for name,want in json.loads(path.read_bytes())['pins'].items():require(sha(ROOT/name)==want,'Frozen incremental builder source changed '+name)
 return sha(path)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--parent',type=Path,required=True);parser.add_argument('--parent-receipt-sha256',required=True);parser.add_argument('--jobs',type=int,choices=range(1,9),default=4);parser.add_argument('--lease-card',type=int,choices=(0,1),default=1);parser.add_argument('--inspect-only',type=Path);parser.add_argument('--leased',action='store_true');a=parser.parse_args()
 if a.inspect_only:
  _,_,proof=inspect_parent(a.parent,a.parent_receipt_sha256);require(not a.inspect_only.exists(),'New inspection output required');a.inspect_only.write_text(json.dumps(proof,indent=2)+'\n',encoding='ascii');return 0
 if not a.leased:os.execv(str(ROOT/'bin/gpu-run'),['gpu-run','--card',str(a.lease_card),sys.executable,__file__,*sys.argv[1:],'--leased'])
 require(os.path.samefile('/proc/self/fd/'+str(8+a.lease_card),'/mnt/vm_8tb/b70/gpu.lock.'+str(a.lease_card)),'Inherited selected compiler owner lease required')
 builder_source_plan_sha256=source_closure()
 parent_plan,child_plan,proof=inspect_parent(a.parent,a.parent_receipt_sha256)
 require(base.output(['git','-C',str(base.SOURCE),'rev-parse','HEAD'])==child_plan['source_revision'] and not base.output(['git','-C',str(base.SOURCE),'status','--porcelain']),'External reviewed source must remain clean')
 for name,want in child_plan['source_file_sha256'].items():require(sha(base.SOURCE/name)==want,'External current source changed '+name)
 require(base.output(['docker','image','inspect',base.IMAGE,'--format','{{.Id}}'])==base.IMAGE,'Exact same compiler image required')
 directory=Path(tempfile.mkdtemp(prefix='strata-native56-incremental55-v2-'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'-',dir='/mnt/vm_8tb/b70/build'))
 (directory/'plan.snapshot.json').write_bytes(CHILD_PLAN.read_bytes());(directory/'controller.py').write_bytes(Path(__file__).read_bytes());(directory/'parent-provenance.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='ascii')
 source=directory/'source';dependency=directory/'ggml-source';build=directory/'build'
 receipt=dict(schema=1,build_strategy='direct55_to56_isolated_incremental_v2',fresh_all_source_compilation=False,parent_provenance_sha256=sha(directory/'parent-provenance.json'),parent_receipt_sha256=a.parent_receipt_sha256,parent_root=str(a.parent),image=base.IMAGE,devices_exposed=False,source_revision=child_plan['source_revision'],ggml_revision=child_plan['ggml']['revision'],source_copy=str(source),ggml_source=str(dependency),build=str(build),plan_snapshot=str(directory/'plan.snapshot.json'),plan_sha256=CHILD_PLAN_SHA,patches=child_plan['patches'],builder_sha256=sha(Path(__file__)),dependency_auditor_sha256=sha(HERE/'native56_incremental_ninja_audit_v1.py'),builder_source_plan_sha256=builder_source_plan_sha256,commands=[],build_rc=1,status='preparing incremental',reused_objects_sha256={},newly_compiled_objects=[],all8_ELFs_deleted_before_relink=False)
 def run(command,log):receipt['commands'].append(command);subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
 try:
  with (directory/'build.log').open('w',encoding='ascii') as log:
   for old,new in ((a.parent/'source',source),(a.parent/'ggml-source',dependency),(a.parent/'build',build)):
    shutil.copytree(old,new,copy_function=shutil.copy2);require(all(p.stat().st_ino!= (old/p.relative_to(new)).stat().st_ino or p.stat().st_dev!=(old/p.relative_to(new)).stat().st_dev for p in new.rglob('*') if p.is_file()),'Isolated copies must not share parent inodes')
   require(tree(source,True)==proof['source_tree'] and tree(dependency,True)==proof['ggml_tree'],'Exact isolated source/dependency copies differ')
   for name,want in proof['object_sha256'].items():require(sha(build/name)==want,'Copied object differs '+name)
   delta=child_plan['patches'][-1];run(['git','-C',str(source),'apply','--check',str(ROOT/delta['path'])],log);run(['git','-C',str(source),'apply',str(ROOT/delta['path'])],log)
   require({k:sha(source/k) for k in child_plan['expected_patched_source_sha256']}==child_plan['expected_patched_source_sha256'],'All74 child source bytes required')
   changed_tree=tree(source,True);require([name for name in changed_tree if changed_tree[name]!=proof['source_tree'][name]]==[CHANGED] and set(changed_tree)==set(proof['source_tree']),'Only authoritative composition body may change')
   for target in child_plan['build_targets']:(build/target).unlink()
   receipt['all8_ELFs_deleted_before_relink']=True
   configure=['cmake','-S','/src/sycl','-B','/build','-G','Ninja']+['-D'+k+'='+v for k,v in sorted(child_plan['cmake_flags'].items())]
   # The shell runs only in the exact compiler image; no DRI or GPU device mount.
   script='\n'.join(['set -Eeo pipefail',"trap 'rc=$?; echo incremental-startup-or-build-failed line=$LINENO rc=$rc >&2; exit $rc' ERR",'test ! -e /dev/dri','if [[ ${SETVARS_COMPLETED:-0} != 1 ]]; then source /opt/intel/oneapi/setvars.sh; fi','icpx --version > /build/incremental-compiler-version.txt','dpkg-query -W > /build/incremental-packages.txt','cmp /build/compiler-version.txt /build/incremental-compiler-version.txt','cmp /build/packages.txt /build/incremental-packages.txt','ninja -C /build -t commands '+shlex.join(child_plan['build_targets'])+' > /build/incremental-parent-commands.txt',shlex.join(configure),'ninja -C /build -t commands '+shlex.join(child_plan['build_targets'])+' > /build/incremental-child-commands.txt','cmp /build/incremental-parent-commands.txt /build/incremental-child-commands.txt','ninja -C /build -t deps > /build/incremental-dependencies.txt','ninja -C /build -n -v '+shlex.join(child_plan['build_targets'])+' > /build/incremental-dryrun.txt','python3 /audit.py', 'ninja -C /build -v '+shlex.join(child_plan['build_targets'])+' -j '+str(a.jobs)])
   receipt['container_script']=script
   run(['docker','run','--rm','--network','none','--user',str(os.getuid())+':'+str(os.getgid()),'-v',str(source)+':/src:ro','-v',str(dependency)+':/ggml:ro','-v',str(build)+':/build','-v',str(directory/'parent-provenance.json')+':/proof.json:ro','-v',str(HERE/'native56_incremental_ninja_audit_v1.py')+':/audit.py:ro',base.IMAGE,script],log)
  after={name:sha(build/name) for name in proof['object_sha256']};require([n for n in after if after[n]!=proof['object_sha256'][n]]==[proof['changed_object']],'Exactly one actual object must be newly compiled; unchanged objects cannot mutate')
  receipt['newly_compiled_objects']=[dict(path=proof['changed_object'],before_sha256=proof['object_sha256'][proof['changed_object']],after_sha256=after[proof['changed_object']])]
  receipt['reused_objects_sha256']={n:w for n,w in after.items() if n!=proof['changed_object']}
  require(len(receipt['reused_objects_sha256'])+1==len(proof['object_sha256']),'Object reuse census differs')
  receipt['binary_sha256']={str(build/n):sha(build/n) for n in child_plan['build_targets']};require(all((build/n).read_bytes()[:4]==b'\x7fELF' for n in child_plan['build_targets']),'All8 newly linked ELF targets required')
  receipt['patched_source_sha256']={n:sha(source/n) for n in child_plan['overlay_files']};receipt['cmake_cache_sha256']=sha(build/'CMakeCache.txt');receipt['incremental_dependency_audit']=json.loads((build/'incremental-audit.json').read_bytes());receipt['ggml_tree']=base.output(['git','-C',str(dependency),'rev-parse','HEAD^{tree}']);receipt.update(build_rc=0,status='incrementally built; 123 objects reused, one body compiled, all8 ELFs relinked; model/math unqualified')
 except BaseException as error:receipt.update(build_rc=1,status='failed incremental; use full builder without reusing this failed tree',error=type(error).__name__+': '+str(error))
 finally:
  try: _,_,closing=inspect_parent(a.parent,a.parent_receipt_sha256);require(closing==proof,'Parent provenance changed during isolated incremental build');receipt['parent_unchanged']=True;require(source_closure()==builder_source_plan_sha256,'Incremental builder closure changed during execution')
  except BaseException as error:receipt.update(build_rc=1,parent_unchanged=False,closing_error=str(error))
  receipt['external_source_unchanged']=(base.output(['git','-C',str(base.SOURCE),'rev-parse','HEAD'])==child_plan['source_revision'] and not base.output(['git','-C',str(base.SOURCE),'status','--porcelain']));receipt['plan_snapshot_unchanged']=sha(directory/'plan.snapshot.json')==CHILD_PLAN_SHA and sha(CHILD_PLAN)==CHILD_PLAN_SHA
  if not receipt['external_source_unchanged'] or not receipt['plan_snapshot_unchanged']:receipt['build_rc']=1
  (directory/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='ascii')
 print(json.dumps(dict(receipt=str(directory/'receipt.json'),build_rc=receipt['build_rc'],build_strategy=receipt['build_strategy'])));return receipt['build_rc']
if __name__=='__main__':raise SystemExit(main())
