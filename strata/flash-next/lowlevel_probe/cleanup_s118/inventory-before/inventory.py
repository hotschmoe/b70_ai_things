from pathlib import Path
import datetime
import json
import os
import subprocess
import time

REPO=Path('/mnt/vm_8tb/github/b70_ai_things')
R=Path('/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014')
OUT=R/'cleanup-inventory-20261016-next-kernel'
records=[]
def call(name,args,timeout=20):
    start=time.monotonic()
    p=subprocess.run(args,cwd=REPO,text=True,capture_output=True,timeout=timeout)
    record={'name':name,'argv':args,'exit':p.returncode,'elapsed_s':time.monotonic()-start,
            'stdout':p.stdout,'stderr':p.stderr}
    if name=='container_state_mounts':
        record['stdout']='OMITTED_FULL_INSPECT: only Id/Name/Image/State/Mounts/Created retained in docker-metadata.json; environment and complete Config intentionally excluded.'
    records.append(record)
    return p
def save(name,j):
    (OUT/name).write_text(json.dumps(j,indent=2,ensure_ascii=True)+'\n',encoding='ascii')

worktrees=call('worktrees',['git','worktree','list','--porcelain'])
dirty=call('dirty',['git','status','--porcelain=v1','--untracked-files=normal'])
ps=call('containers',['docker','ps','-a','--no-trunc','--format','{{json .}}'])
images=call('images',['docker','image','ls','--no-trunc','--format','{{json .}}'])
containers=[json.loads(x) for x in ps.stdout.splitlines()] if ps.returncode==0 else []
image_rows=[json.loads(x) for x in images.stdout.splitlines()] if images.returncode==0 else []
container_details=[]
if containers:
    p=call('container_state_mounts',['docker','container','inspect']+[c['ID'] for c in containers])
    if p.returncode==0:
        for j in json.loads(p.stdout):
            container_details.append({k:j.get(k) for k in ['Id','Name','Image','State','Mounts','Created']})

relative=[
 'greedy-top1-s112-oracle-prep-v2/build/tmp',
 'greedy-top1-s112-oracle-prep-v2/build/cache',
 'greedy-top1-s112-run-v2/tmp',
 'greedy-top1-s112-run-v2/cache',
 'mtp-parallel-argmax-main-s114-build/build/tmp',
 'mtp-parallel-argmax-main-s114-build/build/cache',
 'mtp-parallel-argmax-main-s114-audit/tmp',
 's114-default-isa-addendum/tmp',
]
paths=[R/p for p in relative if (R/p).is_dir() and not (R/p).is_symlink()]
# This exact short list is the entire recursive disk scan. Never scan R itself,
# source/build-sycl-ep, models, other backends or archive/quarantine.
du=call('bounded_tmp_cache_allocated_bytes',['du','-s','-B1','--']+[str(p) for p in paths],timeout=25)
sizes={}
if du.returncode==0:
    for line in du.stdout.splitlines():
        n,p=line.split('\t',1);sizes[p]=int(n)

open_refs=[]
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():continue
    try:
        comm=(proc/'comm').read_text().strip()
        for fd in (proc/'fd').iterdir():
            try:target=os.readlink(fd)
            except (FileNotFoundError,PermissionError,OSError):continue
            for p in paths:
                if target==str(p) or target.startswith(str(p)+'/'):
                    open_refs.append({'pid':int(proc.name),'comm':comm,'fd':fd.name,'target':target})
    except (FileNotFoundError,PermissionError,OSError):pass

entries=[]
for p in paths:
    immediate=[]
    for c in p.iterdir():
        st=c.lstat()
        immediate.append({'name':c.name,'type':'symlink' if c.is_symlink() else 'directory' if c.is_dir() else 'file',
            'lstat_bytes':st.st_size,'allocated_bytes':st.st_blocks*512,'mtime_ns':st.st_mtime_ns})
    mount_refs=[]
    for c in container_details:
        for m in c.get('Mounts') or []:
            source=m.get('Source','')
            if source and (str(p)==source or str(p).startswith(source.rstrip('/')+'/') or source.startswith(str(p)+'/')):
                mount_refs.append({'container':c['Id'],'name':c['Name'],'state':c['State'].get('Status'),'source':source,'destination':m.get('Destination')})
    entries.append({'path':str(p),'allocated_bytes':sizes.get(str(p)),
        'immediate_entries':immediate,'observed_open_fds':[x for x in open_refs if x['target']==str(p) or x['target'].startswith(str(p)+'/')],
        'container_mount_refs':mount_refs,'scope':'Only named tmp/cache subtree; parent experiment outputs/binaries/objects/raw evidence protected.'})

save('worktree-dirty.json',{'worktree_porcelain':worktrees.stdout,'dirty_porcelain':dirty.stdout,
    'verdict':'All existing dirty paths protected; no worktree removal candidate when only the registered main clone exists.'})
save('docker-metadata.json',{'containers':containers,'container_state_mounts':container_details,'images':image_rows,
    'limits':'Reported image Size includes shared layers; no unique/reclaimable-byte claim. Unused/untagged is not proof unreferenced. All existing/current/fallback/pinned/unknown objects retained.'})
save('tmp-cache-inventory.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'scan_roots':entries,'du_exit':du.returncode,'open_fd_scope':'Visible /proc fd symlinks only; permission-limited, no persistent guarantee or hidden memory-mapped-file proof.',
    'protected':'All source/build-sycl-ep, S107/S112/S114/S116 binaries/archives/objects/source/results, all R experiment evidence except explicit expendable temporary subtrees after root approval, models/fallback/archive/quarantine/user dirty paths/current containers.',
    'no_deletion':True})
save('commands.json',records)
print(json.dumps({'status':'READ_ONLY_METADATA_COMPLETE_NO_DELETE','bounded_du_exit':du.returncode,
    'tmp_cache_roots':len(entries),'tmp_cache_allocated_bytes':sum(x['allocated_bytes'] or 0 for x in entries),
    'containers':len(containers),'image_rows':len(image_rows),'visible_open_refs':len(open_refs)}))
