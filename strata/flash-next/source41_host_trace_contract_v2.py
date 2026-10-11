"""Strict LP20/native PID/type checks before unchanged scoped H36 primitive."""
import hashlib,json,math
from pathlib import Path
import collect_decode40_host_trace_v1 as primitive
from serial37_canonical_json_v3 import canonical
require=primitive.original.require;K=primitive.original.KINDS
META_INTS={'schema','pid','request','rid','slotgen','prompt','fresh','produced','events','capacity'}
META_BOOLS={'complete','overflow','thread_error','clock_error'}
META_KEYS=META_INTS|META_BOOLS|{'clock','device_clock','binding'}
EVENT_KEYS={'seq','ns','span','kind','edge','value','dev','lb','le','T','pos','queue','graph'}

def lp20(raw):
    require(type(raw['stdout'])is list and type(raw['LP'])is list and type(raw['output_ids'])is list,'Original typed stdout/LP/token rows required')
    stream=[line for line in raw['stdout']if line.startswith(('T ','LP '))];require(len(stream)==2*len(raw['output_ids'])and raw['LP']==[line for line in stream if line.startswith('LP ')],'Exactly one original LP20 row per actual T required')
    for ordinal,token in enumerate(raw['output_ids']):
        require(type(token)is int and 0<=token<248320 and stream[2*ordinal]=='T '+str(token),'Actual T/LP ordinal/source ID association differs')
        fields=stream[2*ordinal+1].split();require(len(fields)==22 and fields[0]=='LP','Exact LP scalar plus20-entry shape required');require(math.isfinite(float(fields[1])),'Finite token logprob required');ids=[]
        for field in fields[2:]:
            parts=field.split(':');require(len(parts)==2 and parts[0].isdigit(),'Exact LP ID:value entry required');value=int(parts[0]);require(str(value)==parts[0]and 0<=value<248320 and math.isfinite(float(parts[1])),'Typed vocabulary/finite LP value required');ids.append(value)
        require(len(set(ids))==20,'Exactly20 distinct original LP IDs required')
    # Top1 need not equal T: source partial_sort may reorder equal maxima.
    return list(raw['LP'])

def native_namespace(terminal,command):
    host=terminal['HostConfig'];require(type(host.get('PidMode'))is str and host['PidMode']in('','private'),'Actual isolated native PID namespace required')
    require(host.get('Init')is None or host.get('Init')is False,'Original direct exec requires no init wrapper')
    require('--init'not in command and '--pid'not in command and terminal['Config']['Entrypoint']==['/bin/bash']and '--entrypoint'in command and command[command.index('--entrypoint')+1]=='/bin/bash','Actual direct-exec command namespace required')
    require(command[-2]=='-c'and command[-1].startswith('exec 2>&1; source /opt/intel/oneapi/setvars.sh ')and '; exec /build/strata --serve 'in command[-1],'Actual bash PID1 replacement by native executable required')
    return 1

def unique(pairs):
    value={}
    for key,item in pairs:require(key not in value,'Duplicate original trace key');value[key]=item
    return value

def typed_rows(path,pid):
    path=Path(path);require(path.is_file()and not path.is_symlink()and path.stat().st_size<=16<<20,'Bounded regular H36 trace required');payload=path.read_bytes();require(payload.endswith(b'\n')and len(payload)<=16<<20,'Complete bounded original trace lines required');lines=payload.splitlines();require(3<=len(lines)<=32770 and all(len(line)<=4096 for line in lines),'Bounded actual H36 event extent required');rows=[json.loads(line,object_pairs_hook=unique)for line in lines];meta,events,footer=rows[0],rows[1:-1],rows[-1]
    require(type(meta)is dict and set(meta)==META_KEYS and all(type(meta[k])is int for k in META_INTS)and all(type(meta[k])is bool for k in META_BOOLS),'Exact typed original H36 metadata required');require(type(pid)is int and meta['pid']==pid==1,'Actual native PID1 trace association required')
    require(meta['schema']==36 and meta['capacity']==32768 and meta['events']==len(events)and 1<=meta['request']<=4 and 1<=meta['produced']<=64,'Original bounded schema/request/count required')
    for event in events:require(type(event)is dict and set(event)==EVENT_KEYS and all(type(value)is int for value in event.values()),'Exact integer original event fields required')
    require(type(footer)is dict and set(footer)=={'footer','records','write_error'}and type(footer['footer'])is bool and type(footer['records'])is int and type(footer['write_error'])is bool,'Exact typed original H36 footer required')
    return rows,hashlib.sha256(payload).hexdigest()

def collect(paths,log,binding,requests,stages,native_pid):
    before=[]
    for path,raw in zip(paths,requests):
        lp20(raw);rows,digest=typed_rows(path,native_pid);before.append((path,digest));events=rows[1:-1];tokens=[i for i,e in enumerate(events)if e['kind']==K['TEmit']]
        for ordinal,index in enumerate(tokens):
            limit=tokens[ordinal+1]if ordinal+1<len(tokens)else len(events);chunk=events[index+1:limit];read=[e for e in chunk if e['kind']==K['LPReadback']];cpu=[e for e in chunk if e['kind']==K['LPCPU']]
            require(len(read)==len(cpu)==2 and [e['edge']for e in read]==[1,2]and [e['edge']for e in cpu]==[1,2]and read[0]['span']==read[1]['span']and cpu[0]['span']==cpu[1]['span']and index<read[0]['seq']<read[1]['seq']<cpu[0]['seq']<cpu[1]['seq']<limit,'Actual per-token ordered LPReadback/LPCPU source coverage required')
        require(len(tokens)==len(raw['LP']),'Actual token/LP scope roster differs')
    result=primitive.collect(paths,log,binding,requests,stages)
    for path,digest in before:require(hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,'Original typed trace bytes changed during collection')
    result.update(scope='bounded actual source41 HOST decode coverage; optimized batched prompt internals unobserved; durations overlap',native_namespace_pid=1,mandatory_LP20_recollected=True,source41_classifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    return result
