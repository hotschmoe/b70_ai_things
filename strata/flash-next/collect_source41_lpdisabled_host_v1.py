"""Strict LP omission and no LP scopes over unchanged decode-only primitive."""
import hashlib
from pathlib import Path
import source41_host_trace_contract_v2 as strict
require=strict.require;native_namespace=strict.native_namespace

def no_lp(raw):
    require(type(raw['LP'])is list and raw['LP']==[]and not any(line.startswith('LP ')for line in raw['stdout']),'Actual omitted LP request emitted LP rows')

def collect(paths,log,binding,requests,stages,native_pid):
    before=[]
    for path,raw in zip(paths,requests):
        no_lp(raw);rows,digest=strict.typed_rows(path,native_pid);before.append((path,digest));require(not any(event['kind']in(strict.K['LPReadback'],strict.K['LPCPU'])for event in rows[1:-1]),'Actual omitted LP request emitted LPReadback/LPCPU scope')
    result=strict.primitive.collect(paths,log,binding,requests,stages)
    for path,digest in before:require(hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,'Original LPdisabled trace bytes changed')
    result.update(scope='bounded actual source41 HOST decode LPdisabled; optimized prompt internals unobserved; durations overlap',native_namespace_pid=1,LP_protocol_and_scope_absence_recollected=True,LPdisabled_classifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    return result
