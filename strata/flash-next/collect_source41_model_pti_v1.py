"""Model-observed clock/Chrome IDs; no fixture cardinality or physical idle truth."""
import ast,copy,heapq,json,math
from pathlib import Path
import probe_unitrace_indexer40_v5 as original
from source41_pti_raw_budget_v1 import require

def clock_port(expected_device_clock):
    namespace={**vars(original),'EXPECTED_DEVICE_CLOCK':expected_device_clock};node=copy.deepcopy(next(n for n in ast.parse(Path(original.__file__).read_bytes()).body if isinstance(n,ast.FunctionDef)and n.name=='clock_binding'));counts={'chrome_cap':0,'byte_cap':0,'row_cap':0,'positive':0,'observed_count':0}
    for n in ast.walk(node):
        if isinstance(n,ast.BinOp)and isinstance(n.op,ast.LShift)and isinstance(n.left,ast.Constant)and n.left.value==8 and isinstance(n.right,ast.Constant)and n.right.value==20:n.left.value=512;counts['byte_cap']+=1
        if isinstance(n,ast.BinOp)and isinstance(n.op,ast.LShift)and isinstance(n.left,ast.Constant)and n.left.value==64 and isinstance(n.right,ast.Constant)and n.right.value==20:n.left.value=1;n.right.value=30;counts['chrome_cap']+=1
        if isinstance(n,ast.Compare)and isinstance(n.left,ast.Call)and isinstance(n.left.func,ast.Name)and n.left.func.id=='len'and n.comparators and isinstance(n.comparators[0],ast.Constant):
            if n.comparators[0].value==10000:n.comparators[0]=ast.Constant(512<<20);counts['row_cap']+=1
            elif n.comparators[0].value==919:
                if ast.unparse(n.left)=='len(instances)':n.ops=[ast.Gt()];n.comparators=[ast.Constant(0)];counts['positive']+=1
                else:n.comparators=[ast.parse('len(instances)',mode='eval').body];counts['observed_count']+=1
        if isinstance(n,ast.Dict):
            for index,value in enumerate(n.values):
                if isinstance(value,ast.Constant)and value.value==919:n.values[index]=ast.parse('len(instances)',mode='eval').body;counts['observed_count']+=1
    require(counts=={'chrome_cap':2,'byte_cap':2,'row_cap':1,'positive':1,'observed_count':3},'Exact model-specific observed clock count port required')
    exec(compile(ast.fix_missing_locations(ast.Module([node],type_ignores=[])),__file__+'[model-clock-counts]','exec'),namespace);return namespace['clock_binding']

def collect(directory,runtime):
    directory=Path(directory);clock=clock_port(runtime['clock_properties'])(directory,'on');require(clock['process_pid']==1,'Clock writer must join actual direct native PID1')
    files=sorted((directory/'unitrace').glob('clock.*.jsonl'));rows=[original.read_json_line(line)for line in files[0].read_bytes().splitlines()];require(all(len(line)<=4096 for line in files[0].read_bytes().splitlines()),'Actual metadata row bound exceeded')
    instances=[row for row in rows if row['kind']=='instance_normalization'];chrome=next((directory/'unitrace').glob('*.json'));events=original.read(chrome)['traceEvents'];gpu={str(event['args']['id']):event for event in events if event.get('ph')=='X'and event.get('cat')=='gpu_op'}
    require(len(gpu)==len(instances),'Unique original GPU event IDs required');joins=[];gaps=[];commands={}
    sources={};sinks_by_id={};api_threads={};flow_threads={}
    for event in events:
        if event.get('cat','').startswith('Flow_H2D_'):
            require(type(event.get('id'))is int and type(event.get('pid'))is int and type(event.get('tid'))is int and type(event.get('ts'))in(int,float)and math.isfinite(event['ts']),'Typed original flow identity/time required')
            if event.get('ph')=='s'and event['pid']==1:sources.setdefault(event['id'],[]).append(event);flow_threads.setdefault(event['tid'],[]).append(event)
            elif event.get('ph')=='t':sinks_by_id.setdefault(event['id'],[]).append(event)
        elif event.get('ph')=='X'and event.get('cat')=='cpu_op'and event.get('pid')==1 and str(event.get('name','')).startswith('zeCommand'):
            require(type(event.get('tid'))is int and type(event.get('ts'))in(int,float)and type(event.get('dur'))in(int,float)and math.isfinite(event['ts'])and math.isfinite(event['dur'])and event['dur']>=0,'Typed original submit API interval required');api_threads.setdefault(event['tid'],[]).append(event)
    api_by_instance={}
    for tid,flows in flow_threads.items():
        APIs=sorted(api_threads.get(tid,[]),key=lambda e:e['ts']);at=0;active={};ends=[]
        for flow in sorted(flows,key=lambda e:e['ts']):
            stamp=flow['ts']
            while at<len(APIs)and APIs[at]['ts']<=stamp:
                active[at]=APIs[at];heapq.heappush(ends,(APIs[at]['ts']+APIs[at]['dur'],at));at+=1
            while ends and ends[0][0]<stamp:
                _,index=heapq.heappop(ends);active.pop(index,None)
            api_by_instance.setdefault(flow['id'],[]).extend(active.values())
    for row in instances:
        key=str(row['instance_id']);device=gpu[key];require(type(device['pid'])is int and type(device['tid'])is int and type(device['ts'])in(int,float)and type(device['dur'])in(int,float)and math.isfinite(device['ts'])and math.isfinite(device['dur'])and device['dur']>=0,'Typed original GPU event required')
        require(row['device_handle']>0,'Original device ownership required');append_key=(row['command_list_handle'],row['command_id'],row['append_host_ns']);commands.setdefault(append_key,[]).append({'instance_id':row['instance_id'],'graph_command':row['graph_command']})
        source=sources.get(row['instance_id'],[]);sinks=[e for e in sinks_by_id.get(row['instance_id'],[])if e['pid']==device['pid']];api=api_by_instance.get(row['instance_id'],[])
        if len(source)!=1 or len(sinks)!=1 or len(api)!=1:gaps.append({'instance_id':row['instance_id'],'host_sources':len(source),'GPU_sinks':len(sinks),'unique_submit_API_slices':len(api),'graph_command':row['graph_command']})
        joins.append({'instance_id':row['instance_id'],'command_id':row['command_id'],'command_list_handle':row['command_list_handle'],'queue_handle':row['queue_handle'],'device_handle':row['device_handle'],'engine_ordinal':row['engine_ordinal'],'engine_index':row['engine_index'],'graph_command':row['graph_command'],'kernel_name':device.get('name'),'raw_start_ticks':row['raw_kernel_start_ticks'],'raw_end_ticks':row['raw_kernel_end_ticks'],'submit_API':api[0]['name']if len(api)==1 else None})
    return {'clock':clock,'instances':joins,'observed_append_identity_repeats':{str(k):v for k,v in commands.items()if len(v)>1},'kernel_command_id_is_not_unique_append_identity':True,'expected_GPU_instance_ids_derived_from_actual_clock_append_records':True,'fixture_operation_counts_used':False,'flow_gaps':gaps,'observed_submit_flow_coverage_passed':not gaps,'model_source_semantic_operation_coverage_qualified':False,'H36_SYCL_to_LevelZero_mapping_qualified':False,'device_clock_calibration_qualified':False,'physical_idle_qualified':False,'physical_copy_bytes_qualified':False,'uncovered_gaps':['Clock/Chrome records cover observed tracer instances; completeness against all model/injected commands is not proven.','No source-known byte-size record, H36 pointer-to-L0 handle mapping or continuous clock drift bound is available.']}
