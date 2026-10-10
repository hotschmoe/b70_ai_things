"""Purpose-only batch0 frontend protocol identity; original math route unchanged."""
import ast,hashlib
from pathlib import Path

SOURCE_SHA='3166f70fa013b905adf713dba404a2eb91a4f3381842dafb542af98cf978d241'
def require(ok,message):
 if not ok:raise ValueError(message)

def method_source(path):
 raw=Path(path).read_bytes();require(hashlib.sha256(raw).hexdigest()==SOURCE_SHA,'Exact consumed source39 frontend required');source=raw.decode('utf-8');tree=ast.parse(source);classes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='StrataEngine'];require(len(classes)==1,'Exact production StrataEngine required');nodes=[n for n in classes[0].body if isinstance(n,ast.FunctionDef) and n.name=='generate'];require(len(nodes)==1,'Exact original serial generate method required');node=nodes[0];return '\n'.join(source.splitlines()[node.lineno-1:node.end_lineno])+'\n'

def adapt(source):
 """Only original request-identity helper/header keys, never synthetic input IDs."""
 old='        self.progress, self.progress_ms, self.reused = None, 0, 0\n'
 addition="""        if not getattr(self,'strict_batch_identity',False):
            raise ValueError('Purpose batch0 identity requires actual protocol2 capability')
        strict_batch_request(sampling,embeddings)
        self._begin_request_identity()
"""
 require(source.count(old)==1 and source.count('self.sampling_keys(sampling or {})')==2,'Exact original batch0 method/header shape changed')
 return source.replace(old,addition+old).replace('self.sampling_keys(sampling or {})','self._request_keys(sampling or {})')

def undo(source):
 old='        self.progress, self.progress_ms, self.reused = None, 0, 0\n';addition="""        if not getattr(self,'strict_batch_identity',False):
            raise ValueError('Purpose batch0 identity requires actual protocol2 capability')
        strict_batch_request(sampling,embeddings)
        self._begin_request_identity()
"""
 require(source.count(addition+old)==1 and source.count('self._request_keys(sampling or {})')==2,'Exact purpose modifications changed');return source.replace(addition+old,old).replace('self._request_keys(sampling or {})','self.sampling_keys(sampling or {})')

def install(server,source_path):
 original=method_source(source_path);adapted=adapt(original);require(undo(adapted)==original,'Original body must remain exact outside identity changes')
 namespace=dict(vars(server));exec(compile('if True:\n'+adapted,str(source_path)+'[purpose-batch0-owned-RID-only]','exec'),namespace);server.StrataEngine.generate=namespace['generate']
 return {'original_source_sha256':SOURCE_SHA,'original_method_sha256':hashlib.sha256(original.encode()).hexdigest(),'purpose_method_sha256':hashlib.sha256(adapted.encode()).hexdigest(),'only_actual_request_identity_and_header_changed':True,'forced_dispatch_or_math_change':False,'actual_runtime_or_full_cache_qualified':False}
