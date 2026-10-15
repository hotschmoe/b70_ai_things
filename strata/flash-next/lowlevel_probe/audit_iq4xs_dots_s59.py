#!/usr/bin/env python3
"""S59 host-only capture inventory. Does not establish physical GPU dispatch."""
import argparse,hashlib,json,re,subprocess,sys
from pathlib import Path

ARMS={'A1':1,'B1':2,'B2':2,'A2':1}
LEAF='9755445400226b7108bf98552411fd603d9489fa2288c42425902a8dfcf61797'

def require(ok,msg):
    if not ok:raise ValueError(msg)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def parse(path):
    nodes={};edges=[]
    for line in path.read_text(encoding='ascii').splitlines():
        line=line.strip()
        if not line or line=='digraph dot {' or line=='}':continue
        edge=re.fullmatch(r'"([^"\n]+)"\s*->\s*"([^"\n]+)"\s*;?',line)
        if edge:edges.append(tuple(edge.groups()));continue
        node=re.fullmatch(r'"([^"\n]+)"\s*\[(.*)\]\s*;',line)
        require(node is not None,f'{path}: unrecognized DOT statement')
        nodeid,attrs=node.groups();require(nodeid not in nodes,f'{path}: duplicate node ID')
        label=re.search(r'(?:^|,\s*)label="((?:\\.|[^"\\])*)"',attrs)
        require(label is not None,f'{path}: missing node label')
        label=json.loads('"'+label[1]+'"')
        fields={}
        args={}
        for part in label.splitlines():
            if part.startswith(('ID = ','TYPE = ','NAME = ')):
                k,v=part.split(' = ',1);fields[k]=v.strip()
            elif re.match(r'^\d+\)',part):
                arg=re.fullmatch(r'(\d+)\) Type: (Pointer|STD_Layout) Ptr: (0x[0-9a-fA-F]+)(?:\((0x[0-9a-fA-F]+)\))?',part.strip())
                require(arg is not None,f'{path}: unsupported argument format')
                idx,kind,storage,value=arg.groups();idx=int(idx)
                require(idx not in args,f'{path}: duplicate argument')
                require((kind=='Pointer')==(value is not None),f'{path}: missing pointer VALUE or opaque scalar exposed')
                args[idx]={'kind':kind,'value':value.lower() if value else None}
        require(fields.get('ID')==nodeid and fields.get('TYPE')=='CGExecKernel',f'{path}: unsupported node type/ID')
        require(bool(fields.get('NAME')),f'{path}: missing kernel name')
        nodes[nodeid]={'mangled_name':fields['NAME'],'args':args}
    require(nodes,f'{path}: no kernel nodes')
    require(len(edges)==len(set(edges)),f'{path}: duplicate edges')
    incoming={n:[] for n in nodes};outgoing={n:[] for n in nodes}
    for a,b in edges:
        require(a in nodes and b in nodes and a!=b,f'{path}: invalid edge endpoint/self edge')
        outgoing[a].append(b);incoming[b].append(a)
    starts=[n for n in nodes if not incoming[n]];ends=[n for n in nodes if not outgoing[n]]
    require(len(starts)==len(ends)==1,f'{path}: not one start/end')
    require(all(len(incoming[n])==(n!=starts[0]) and len(outgoing[n])==(n!=ends[0]) for n in nodes),f'{path}: branch/join/disconnected chain')
    order=[];n=starts[0]
    while True:
        require(n not in order,f'{path}: cycle');order.append(n)
        if not outgoing[n]:break
        n=outgoing[n][0]
    require(len(order)==len(nodes) and len(edges)==len(nodes)-1,f'{path}: disconnected or extra nodes')
    return nodes,edges,order

def classify(name,rpw):
    if '::native_quantize_q8_1_kernel_b480d2' in name:
        require('strata::kernels::native_quantize_q8_1(' in name,'wrong quantizer owner')
        return 'quantize'
    require(re.search(r'launch_wide<[^>]*WideIQ4XS,\s*6>',name) is not None,'wrong format/column template')
    require('dpct_kernel_scalar<6>' in name,'missing C6 kernel scalar')
    if rpw==1:
        require('::native_mmvq_wide,' in name and 'native_mmvq_wide_rpw' not in name,'wrong RPW1 kernel')
        require('dpct_kernel_scalar<2>' not in name and 'dpct_kernel_scalar<4>' not in name,'unexpected RPW scalar')
    else:
        require('::native_mmvq_wide_rpw,' in name,'wrong RPW2 kernel')
        require(re.search(r'dpct_kernel_scalar<6>,\s*dpct_kernel_scalar<2>\s*>',name) is not None,'wrong RPW scalar template order')
        require('dpct_kernel_scalar<4>' not in name,'RPW4 unexpectedly captured')
    return 'projection'

def audit(root):
    expected={root/a/f'C6-{g}.dot' for a in ARMS for g in ['single','batch20']}
    observed={p for a in ARMS for p in (root/a).glob('*.dot')}
    require(observed==expected,'missing/additional DOT files: '+str(sorted(str(x) for x in observed^expected)))
    parsed={p:parse(p) for p in sorted(expected)}
    unique=sorted({n['mangled_name'] for nodes,_,_ in parsed.values() for n in nodes.values()})
    cp=subprocess.run(['c++filt','-t'],input='\n'.join(unique)+'\n',text=True,capture_output=True,check=True)
    decoded=cp.stdout.splitlines();require(len(decoded)==len(unique),'demangler result count')
    demangled=dict(zip(unique,decoded))
    require(all(a!=b for a,b in demangled.items()),'unresolved mangled kernel name')
    result={};bindings_by_arm={}
    for arm,rpw in ARMS.items():
        result[arm]={};arm_bindings=None
        for graph,calls in [('single',1),('batch20',20)]:
            p=root/arm/f'C6-{graph}.dot';nodes,edges,order=parsed[p]
            require(len(nodes)==2*calls,f'{arm}/{graph}: node count')
            kinds=[classify(demangled[nodes[n]['mangled_name']],rpw) for n in order]
            require(kinds==['quantize','projection']*calls,f'{arm}/{graph}: exact directed alternation failed')
            for i in range(calls):
                q=nodes[order[2*i]]['args'];m=nodes[order[2*i+1]]['args']
                require(set(q)=={0,1,2} and set(m)=={0,1,2,3,4},f'{arm}/{graph}: unexpected closure ABI')
                require([q[j]['kind'] for j in range(3)]==['Pointer','Pointer','STD_Layout'],f'{arm}: quantizer arg types')
                require([m[j]['kind'] for j in range(5)]==['Pointer','Pointer','Pointer','STD_Layout','STD_Layout'],f'{arm}: projection arg types')
                require(q[1]['value']==m[1]['value'],f'{arm}: projection does not read preceding Q8 base')
                binding={'float_input':q[0]['value'],'Q8_input_output':q[1]['value'],'weights':m[0]['value'],'projection_output':m[2]['value']}
                require(len(set(binding.values()))==4 and all(int(v,16)!=0 for v in binding.values()),f'{arm}: aliased/null buffers')
                if arm_bindings is None:arm_bindings=binding
                require(binding==arm_bindings,f'{arm}: bindings changed across repeats/single/batch captures')
            result[arm][graph]={'file':str(p),'sha256':sha(p),'rpw':rpw,'nodes':len(nodes),'edges':len(edges),'calls':calls,'exact_chain':True,'ordered_node_ids':order,'alternating_quantize_projection':True,'unique_demangled_names':sorted({demangled[n['mangled_name']] for n in nodes.values()}),'opaque_scalar_arguments':{'quantizer':[2],'projection':[3,4]},'pointer_bindings_constant':True}
        bindings_by_arm[arm]=arm_bindings
    identity=root/'identity.json';require(identity.is_file(),'missing run identity')
    identity_record=json.loads(identity.read_text())
    require((identity_record['ggml_type'],identity_record['N'],identity_record['M'],identity_record['C'])==(23,2560,10240,[6]),'identity/source shape mismatch')
    require(identity_record['source_sha256']==LEAF,'unexpected leaf source identity')
    return {'status':'PASS_CAPTURE_CHAIN_AND_BINDINGS','raw_run':str(root),'identity_sha256':sha(identity) if identity.exists() else None,'source_shape_binding':'N2560/M10240 from pinned leaf/runtime identity; opaque DOT scalar values NOT decoded','identity_present':identity_record is not None,'arms':result,'same_arm_single_batch_pointer_values':bindings_by_arm,'pointer_policy':'parenthesized device pointer VALUES, never host argument storage addresses; no cross-process pointer-equality gate','total_DOTs':8,'total_kernel_nodes':168,'total_quantizers':84,'total_projections':84,'total_edges':160,'proof_scope':'actual modifiable-graph capture metadata only; not independent GPU dispatch/execution count/ISA/resources/physical traffic or numerical/health/timing qualification'}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('raw_run',type=Path);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    require(not a.output.exists(),'preserve existing audit output')
    try:report=audit(a.raw_run.resolve());rc=0
    except Exception as e:report={'status':'FAIL_CAPTURE_AUDIT','error':type(e).__name__+': '+str(e),'proof_scope':'no capture/dispatch acceptance; inspect actual format without rerun'};rc=2
    with a.output.open('x',encoding='ascii') as f:json.dump(report,f,indent=2,ensure_ascii=True);f.write('\n')
    print(json.dumps({'status':report['status'],'output':str(a.output)},ensure_ascii=True));return rc
if __name__=='__main__':raise SystemExit(main())
