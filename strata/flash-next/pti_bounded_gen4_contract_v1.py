"""Executable cold GEN4 recipe constructor/coverage; not runtime admission."""
import copy,json,hashlib,argparse
from pathlib import Path

def require(ok,msg):
    if not ok:raise ValueError(msg)
def construct(recipe,inputs):
    require(type(recipe)is dict and type(inputs)is dict,'Typed admitted recipe/inputs required')
    for key,size in(('short',54),('prefill_about512',524)):
        require(type(inputs.get(key))is list and len(inputs[key])==size and all(type(x)is int and x>=0 for x in inputs[key]),'Exact useful prompt IDs/count required '+key)
    result=copy.deepcopy(recipe)
    require(type(result.get('requests'))is list and len(result['requests'])==4,'Original matched four-request recipe required')
    for i,key in enumerate(('short','prefill_about512','prefill_about512','short')):
        old=result['requests'][i]
        require(old['ordinal']==i+1 and old['ids']==inputs[key] and old['fresh']==1 and old['pin']is None and old['max_new']==64,'Actual original GEN64 matched intake differs')
        old['max_new']=4
    result.update(bounded_window_schema=1,capture_kind='cold fresh four GEN4 requests',warm_process_decode_qualified=False,selected_request_count=4,maximum_total_output_tokens=16,clean_serving_latency_qualified=False)
    if result.get('PTI_enabled')is True:
        result['native_env']['UNITRACE_CLOCK_QUERY_SAMPLE_512']='1'
    else:
        require('UNITRACE_CLOCK_QUERY_SAMPLE_512'not in result['native_env'],'OFF cannot inherit sampling flag')
    return result

def sampling_summary(rows):
    require(type(rows)is list and len(rows)==6,'Exactly six close summaries required')
    for site,row in enumerate(rows,1):
        names=('query_site','sample_every','original_query_calls','selected_brackets','completed_brackets','suppressed_brackets')
        require(all(type(row.get(n))is int and row[n]>=0 for n in names),'Exact typed counter fields required')
        seen=row['original_query_calls'];expected=seen if site<=2 else (seen+511)//512
        require(row['query_site']==site and row['sample_every']==(1 if site<=2 else 512) and row['selected_brackets']==row['completed_brackets']==expected and row['suppressed_brackets']==seen-expected,'Original query/sample/count closure differs')
    return {'query_brackets_sampled':True,'all_operation_normalizations_required':True,'every_operation_submit_pair_bracketed':False,'clock_alignment_qualified':False}

def main():
    p=argparse.ArgumentParser();p.add_argument('--recipe',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    require(not a.output.exists() and a.output.parent.resolve()==a.output.parent,'New canonical output required')
    r=json.loads(a.recipe.read_bytes());i=json.loads(a.inputs.read_bytes());result=construct(r,i)
    result['original_recipe_sha256']=hashlib.sha256(a.recipe.read_bytes()).hexdigest();result['original_inputs_sha256']=hashlib.sha256(a.inputs.read_bytes()).hexdigest()
    a.output.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n',encoding='ascii')
if __name__=='__main__':main()

def audit_sampled_rows(rows):
    """Streaming row gate; caller retains original full clock/Chrome joins."""
    queries={s:0 for s in range(1,7)};bridges=dict(queries);summaries=[];last=0;pid=None;instances=set();final=False
    for row in rows:
        require(type(row)is dict and type(row.get('sequence'))is int and row['sequence']==last+1,'Contiguous complete clock sequence required');last+=1
        require(type(row.get('pid'))is int and row['pid']>0,'Typed clock owner required')
        if pid is None:pid=row['pid']
        require(row['pid']==pid and not final,'Owner changed or rows after final bridge')
        kind=row.get('kind')
        if kind=='global_timestamp_query_bracket':
            s=row.get('query_site');require(type(s)is int and s in queries and not summaries,'Query after sampling closure or wrong site');queries[s]+=1
        elif kind=='host_clock_bridge':
            s=row.get('site');require(type(s)is int,'Typed bridge site required')
            if s==2:require(len(summaries)==6,'Normal close requires exact summaries');final=True
            elif 101<=s<=106:bridges[s-100]+=1
            else:require(s==0 or s==1,'Unknown bridge site')
        elif kind=='query_sampling_summary':summaries.append(row)
        elif kind=='instance_normalization':
            instance=row.get('instance_id');require(type(instance)is int and instance>0 and instance not in instances,'Unique actual normalization ID required');instances.add(instance)
        else:require(kind in('host_clock_origin','device_origin'),'Unknown clock row kind')
    require(final,'Normal final close bridge required');result=sampling_summary(summaries)
    for r in summaries:require(queries[r['query_site']]==bridges[r['query_site']]==r['completed_brackets'],'Actual emitted query/bridge count differs')
    result.update(pid=pid,rows=last,instance_normalizations=len(instances),query_brackets=queries)
    return result
