"""Exact diagnostic recipes shared by producer and read-only consumer."""
import os,shlex
from pathlib import Path
import c1_serve_controller_combined_v140_v3 as c
from prepare_decode40_host_profile_v1 import require


def mount_binding(actual,expected):
    require(type(actual)is list and type(expected)is list and len(actual)==len(expected),'Exact mount count/list required')
    def destinations(rows):
        parsed=[r.rsplit(':',2)for r in rows]
        return [r[-2]if len(r)==3 else r[-1]for r in parsed]
    require(len(set(destinations(actual)))==len(actual)and len(set(destinations(expected)))==len(expected),'Duplicate mount destination forbidden')
    require(sorted(actual)==sorted(expected),'Exact mount bijection differs')


def native_command(directory,m,plan,pid,binding):
    directory=Path(directory);root=c.REPO;name='b70-h36-'+str(pid)+'-'+plan['mode'].lower()
    cmd=['docker','run','-i','--name',name,'--label','b70.decode40.host='+binding,'--network','none','--device','/dev/dri','--user','1000:1000','--memory','105g','--memory-swap','105g','--group-add',str(os.stat('/dev/dri/renderD128').st_gid),'--group-add',str(os.stat('/dev/dri/card0').st_gid),'--workdir','/src','--entrypoint','/bin/bash']
    lock=c.read(root/'strata/flash-next/model-lock.json')
    for host,target in [(Path(m['executable']).parent,'/build'),(Path(m['source']),'/src'),(Path(m['pack']),'/pack'),(root/lock['destination'],'/model'),(directory,'/results')]:cmd+=['-v',str(host)+':'+target+(''if target=='/results'else':ro')]
    for key,value in sorted(plan['native_env'].items()):cmd+=['-e',key+'='+value]
    cmd += [m['runtime']['image'],'-c','source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; exec '+shlex.join(plan['native_argv'])]
    return name,cmd


def tokenizer_command(out,m,pid,binding,actor_path):
    root=c.REPO;here=root/'strata/flash-next';out=Path(out);source=Path(m['source']);pack=Path(m['pack'])/'tokenizer';name='b70-h36-tokenizer-'+str(pid)
    cmd=['docker','run','--name',name,'--label','b70.decode40.host='+binding,'--network','none','--read-only','--memory','512m','--memory-swap','512m','--cpus','1','--pids-limit','128','--user',str(os.getuid())+':'+str(os.getgid()),'--workdir','/','--entrypoint','/usr/bin/env']
    for host,target in [(Path(actor_path),'/harness/strata/flash-next/actor.py'),(here/'qualify_api_positive_overlap_cpu_screen_v3.py','/harness/strata/flash-next/qualify_api_positive_overlap_cpu_screen_v3.py'),(source/'tools/strata_tokenizer.py','/native-tools/strata_tokenizer.py'),(source/'serve/frontend.py','/native-serve/frontend.py'),(pack,'/tokenizer'),(out,'/results')]:cmd+=['-v',str(host)+':'+target+(''if target=='/results'else':ro')]
    cmd += [m['runtime']['image'],'-i','PATH=/usr/bin:/bin','LANG=C','LC_ALL=C','PYTHONDONTWRITEBYTECODE=1','/opt/b70-c1-python/bin/python','/harness/strata/flash-next/actor.py','--inside-tokenizer','/results/tokenizer-plan.json','/results/tokenizer-fixtures.json']
    return name,cmd


def inspection(obj,name,cmd,image,binding,metadata):
    require(obj['Name']=='/'+name and obj['Config']['Image']==image and obj['Config']['Labels'].get('b70.decode40.host')==binding,'Refusing foreign diagnostic container')
    require(type(obj.get('RestartCount'))is int and obj['RestartCount']==0,'Exact original zero restart count required');hc=obj['HostConfig'];cfg=obj['Config'];require(hc.get('AutoRemove')is False and hc.get('Privileged')is False and hc.get('NetworkMode')=='none','Inspectable unprivileged isolated network required')
    mount_binding(hc.get('Binds'),[cmd[i+1]for i,x in enumerate(cmd)if x=='-v'])
    require(cfg['Cmd']==cmd[cmd.index(image)+1:]and cfg['Entrypoint']==[cmd[cmd.index('--entrypoint')+1]],'Exact entrypoint/command required')
    require(cfg['User']==cmd[cmd.index('--user')+1]and cfg['WorkingDir']==cmd[cmd.index('--workdir')+1],'Exact user/workdir required')
    size=(512<<20)if metadata else 105*(1<<30)
    require(type(hc.get('Memory'))is int and hc['Memory']==size and type(hc.get('MemorySwap'))is int and hc['MemorySwap']==size and hc.get('ReadonlyRootfs')is metadata,'Exact memory/noswap/read-only recipe required')
    require(not hc.get('DeviceRequests'),'No GPU device requests permitted')
    groups=hc.get('GroupAdd')or[];devices=hc.get('Devices')or[]
    if metadata:
        require(not devices and not groups and type(hc.get('NanoCpus'))is int and hc['NanoCpus']==1000000000 and type(hc.get('PidsLimit'))is int and hc['PidsLimit']==128,'Metadata CPU-only resource recipe required')
    else:
        expected=[cmd[i+1]for i,x in enumerate(cmd)if x=='--group-add']
        require(sorted(groups)==sorted(expected)and len(groups)==len(expected)and len(devices)==1 and devices[0]['PathOnHost']==devices[0]['PathInContainer']=='/dev/dri'and devices[0]['CgroupPermissions']=='rwm','Exact native group/device mapping required')
    expected=[cmd[i+1]for i,x in enumerate(cmd)if x=='-e'];observed=cfg['Env']
    require(all(sum(y.split('=',1)[0]==x.split('=',1)[0]for y in observed)==1 and x in observed for x in expected),'Actual diagnostic environment differs or duplicates')
    return obj
