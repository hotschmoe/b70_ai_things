"""Two fresh same-prompt GEN4 rows; warmed graph identity is observed, not assumed."""
import ast,copy
from pathlib import Path
import source41_model_cli_contract_v1 as old
import source41_lpdisabled_contract_v1 as omission
from serial37_canonical_json_v3 import canonical
require=old.require

def requests(rows,expected):
    node=copy.deepcopy(next(n for n in ast.parse(Path(old.__file__).read_bytes()).body if isinstance(n,ast.FunctionDef)and n.name=='requests'))
    changed={'count':0,'LP':0}
    for n in ast.walk(node):
        if isinstance(n,ast.Compare)and ast.unparse(n)=='len(rows) == len(expected) == 4':n.comparators[-1]=ast.Constant(2);changed['count']+=1
        if isinstance(n,ast.Constant)and n.value=='GEN %d fresh=1 ckpt=1 temperature=0 logprobs=20 ':n.value='GEN %d fresh=1 ckpt=1 temperature=0 ';changed['LP']+=1
    require(changed=={'count':1,'LP':1},'Exact two-request LPdisabled contract port required')
    ns=dict(vars(old));exec(compile(ast.fix_missing_locations(ast.Module([node],type_ignores=[])),__file__+'[two-current-fresh-GEN4]','exec'),ns);views=ns['requests'](rows,expected)
    for r in rows:require(r['raw']['LP']==[]and not any(x.startswith('LP ')for x in r['raw']['stdout']),'No LP scope can enter warm window')
    require(canonical(views[0])==canonical(views[1]),'Actual warmup and selected fresh outputs differ');return views

def equivalence(left,right):
    require(canonical(left)==canonical(right)and len(left)==len(right)==2,'Actual matched warmup/selected OFFON output differs')
    return dict(H36_OFF_ON_equivalence_passed=True,scope='two actual same-prompt fresh GEN4 rows; graph warmup and selected window; no serving speed authority')

def graph_reuse(warm_rows,selected_rows,K):
    warm=warm_rows[1:-1];selected=selected_rows[1:-1]
    def keys(events,kind,edge):return {(e['dev'],e['lb'],e['le'],e['T'],e['queue'],e['graph'])for e in events if e['kind']==K[kind]and e['edge']==edge}
    oldgraphs=keys(warm,'ReplaySubmit',1);newgraphs=keys(selected,'ReplaySubmit',1)
    require(oldgraphs and newgraphs and newgraphs<=oldgraphs and all(q>0 and g>0 for *_,q,g in newgraphs),'Actual selected replay must use warmup graph/queue IDs')
    require(not any(e['kind']in(K['GraphCreate'],K['GraphPreexisting'],K['GraphFinalize'],K['Capture'])for e in selected),'Selected request recreated or first-observed a graph')
    hits=keys(selected,'GraphHit',0);require(newgraphs<=hits,'Selected replay lacks original warmed graph-hit record')
    return dict(actual_same_process_graph_reuse=True,warmup_graph_roster=sorted(oldgraphs),selected_graph_roster=sorted(newgraphs),conversation_state_reuse_claimed=False,raw_device_operation_mapping_qualified=False)
