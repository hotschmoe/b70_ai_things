"""Explicit omission-only native request port; never logprobs=0."""
import ast,copy
from pathlib import Path
from types import SimpleNamespace
import native_source41_cli_protocol_v2 as protocol
import source41_model_cli_contract_v1 as contract
from serial37_canonical_json_v3 import canonical
require=contract.require

def request_ports():
    original=protocol.Commands;namespace=dict(vars(__import__(original.__module__)))
    tree=ast.parse(Path(namespace['__file__']).read_bytes());command=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='request_command');command=copy.deepcopy(command);changed=0
    for node in ast.walk(command):
        if isinstance(node,ast.Constant)and node.value=='ckpt=1 temperature=0 logprobs=20 ':node.value='ckpt=1 temperature=0 ';changed+=1
    require(changed==1,'Exactly one original command logprobs omission required')
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='Protocol');request=copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name=='request'))
    exec(compile(ast.fix_missing_locations(ast.Module([command,request],type_ignores=[])),__file__+'[explicit-omitted-request]','exec'),namespace)
    return namespace
_REQUEST=request_ports()
class Protocol(protocol.Protocol):request=_REQUEST['request']

def row_ports():
    node=copy.deepcopy(next(n for n in ast.parse(Path(contract.__file__).read_bytes()).body if isinstance(n,ast.FunctionDef)and n.name=='requests'));changed=0
    for item in ast.walk(node):
        if isinstance(item,ast.Constant)and item.value=='GEN %d fresh=1 ckpt=1 temperature=0 logprobs=20 ':item.value='GEN %d fresh=1 ckpt=1 temperature=0 ';changed+=1
    require(changed==1,'Exactly one original validator command omission required');namespace=dict(vars(contract));exec(compile(ast.fix_missing_locations(ast.Module([node],type_ignores=[])),__file__+'[explicit-omitted-row]','exec'),namespace);return namespace['requests']
_ROWS=row_ports()

def requests(rows,expected):
    result=_ROWS(rows,expected)
    for row in rows:
        raw=row['raw'];require(type(raw['LP'])is list and raw['LP']==[]and not any(line.startswith('LP ')for line in raw['stdout']),'LPdisabled cannot emit any LP protocol line')
    return result

def equivalence(left,right):
    contract.equivalence(left,right)
    return {'H36_OFF_ON_equivalence_passed':True,'scope':'source41mode0 four bounded fresh LPdisabled native host control only'}
