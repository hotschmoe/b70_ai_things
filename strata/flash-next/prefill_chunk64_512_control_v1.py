#!/usr/bin/env python3
"""Explicit single-B70 startup chunk control over unchanged source40 math."""
import ast
import re
import time
from pathlib import Path
import matched_geometry_c140_diagnostic_v3 as base

ROOT=base.ROOT
DELTA=ROOT/'strata/flash-next/prefill-chunk64-512-registry-proposal-v1.yaml'


def profile(name):
    base.original.require(name in ('chunk64','chunk512'),'Exactly one-card chunk64/512 controls supported')
    p=base.profile('matched-one');chunk=64 if name=='chunk64'else 512
    p['prefill']=chunk
    p['alias']=p['alias'].replace('prefill64','prefill'+str(chunk)).replace('matchedgeometry1','chunkcontrol1')
    return p


def expected_node(node):
    changed=0
    for n in ast.walk(node):
        if isinstance(n,ast.Tuple)and len(n.elts)==2 and isinstance(n.elts[0],ast.Constant)and n.elts[0].value=='--prefill':
            base.original.require(isinstance(n.elts[1],ast.Constant)and n.elts[1].value==64,'Exact prior chunk64 constructor required')
            n.elts[1]=ast.Subscript(value=ast.Name(id='p',ctx=ast.Load()),slice=ast.Constant(value='prefill'),ctx=ast.Load());changed+=1
    base.original.require(changed==1,'Exactly one declared constructor chunk change')
    return node


def ports():
    namespace=dict(vars(base));namespace.update(__file__=__file__,__name__=__name__,profile=profile,DELTA=DELTA)
    original_tree=ast.parse(Path(base.original.__file__).read_text())
    old_names=('prepare','validate_prepared','metadata_admission_gate','launch','screen')
    old=[base.derive(n)for n in original_tree.body if isinstance(n,ast.FunctionDef)and n.name in old_names]
    exec(compile(ast.fix_missing_locations(ast.Module(body=old,type_ignores=[])),__file__,'exec'),namespace)
    namespace['_VALIDATE']=namespace['validate_prepared']
    tree=ast.parse(Path(base.__file__).read_text());names=('same','runtime_gate','expected_config','config_gate','predevice_metadata',
        'one_card_prerequisite_metadata','prelease_admission','registry_association','registry_gate','metadata_snapshot','snapshot_equal','validate_prepared','main')
    nodes=[]
    for node in tree.body:
        if not isinstance(node,ast.FunctionDef)or node.name not in names:continue
        if node.name=='expected_config':node=expected_node(node)
        if node.name in ('main','registry_association'):
            count=0
            for n in ast.walk(node):
                if isinstance(n,ast.Tuple)and [getattr(x,'value',None)for x in n.elts]==['matched-one','matched-pair']:
                    n.elts=[ast.Constant(value='chunk64'),ast.Constant(value='chunk512')];count+=1
            base.original.require(count==1,'Exactly one single-card profile/registry choice route')
        nodes.append(node)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),__file__,'exec'),namespace)
    namespace['_PORTS']=namespace
    return namespace


_PORTS=ports()
for _name in ('validate_prepared','prelease_admission','one_card_prerequisite_metadata','registry_association','registry_gate','expected_config','config_gate'):
    globals()[_name]=_PORTS[_name]


def memory_admission(out,metadata):
    """Actual startup observations, never a physical spill/traffic measurement."""
    path=Path(out)/'server-engine.log'
    base.original.require(path.stat().st_size<=16<<20,'Bounded original startup log required')
    with path.open('rb')as f:raw=f.read((16<<20)+1)
    base.original.require(len(raw)<=16<<20,'Bounded actual startup bytes required');text=raw.decode('ascii')
    base.original.require('STRATA_PREFILL_FIRST'not in _PORTS['expected_config'](metadata['profile'],metadata['runtime'],metadata['port'])['env'],
                          'Unchanged absent prefill-first override required')
    free=re.findall(r'strata serve: (\d+) MiB of VRAM free with everything loaded\n',text)
    slots=re.findall(r'strata generate: expert cache (\d+) slots,',text)
    mirrors=re.findall(r'stage mirror: layers0-47 missing(\d+) mirrored(\d+) bytes(\d+) ',text)
    base.original.require(len(free)==len(slots)==len(mirrors)==1,'Unique actual loaded VRAM/cache/mirror startup required')
    resident=int(slots[0]);missing,mirrored,mirrorbytes=map(int,mirrors[0])
    base.original.require(int(free[0])>=1024,'Preregistered1024MiB startup headroom required; no reserve lowering')
    base.original.require(0<resident<24576 and missing==mirrored==24576-resident and mirrorbytes>0,
                          'Actual complete single-stage VRAM/mirror coverage required')
    base.original.require('the prompt path allocates its own buffers'in text and not re.search(r'trying a smaller expert cache|trying a \d+-token chunk| - LOW:',text),
                          'No borrowing/capacity fallback/low-memory diagnostic allowed')
    value={'schema':1,'epoch':time.time(),'log_path':str(path),'log_prefix_bytes':len(raw),'log_prefix_sha256':base.original.hashlib.sha256(raw).hexdigest(),
           'observed_free_VRAM_MiB':int(free[0]),'resident_experts':resident,'mirror_experts':mirrored,'mirror_bytes':mirrorbytes,
           'prefill_first_override_absent':True,'resolved_prefill_first_default':0,'startup_sample_only':True,
           'physical_spill_observed':False,'dynamic_expert_transfer_bytes_observed':False,'quant_or_MTP_changed':False}
    base.original.write(Path(out)/'chunk-memory-admission.json',value)
    return value


if __name__=='__main__':_PORTS['main']()
