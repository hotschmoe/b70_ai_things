"""Explicit actor-injected NUM33 software collector, no __main__ import authority."""
from types import FunctionType
import re
import native62_hc_rms_raw_v1 as prior

def ports(active_actor):
 # Importing prior supplies immutable software code only. Its canonical actor
 # is never used for current runtime SDK/configuration authority.
 require=active_actor.require;c=active_actor.c;sdk=active_actor.parent.fresh
 require(sdk is not None and sdk.SDK is not None,'Explicit actual executing actor SDK required')
 def field_contract():
  p=sdk.SDK/'source/sycl/include/strata/core/layer0_numerical_contract.hpp';want=c.read(sdk.PLAN)['expected_patched_source_sha256']['sycl/include/strata/core/layer0_numerical_contract.hpp'];require(c.sha(p)==want,'Explicit actual NUM33 contract changed');rows=re.findall(r'\{"([^"]+)","([^"]+)",(\d+)\}',p.read_text());require(len(rows)==33 and sum(int(r[2])for r in rows)==7023048,'Original33 field quota required');return [{'name':n,'encoding':e,'bytes':int(b),'observed':True,'provenance':'DERIVED_gpu_native_exp_from_actual_gate_up'if i in(25,29)else'actual_buffer'}for i,(n,e,b)in enumerate(rows)]
 ns={**prior.collect.__globals__,'actor':active_actor,'field_contract':field_contract}
 f=prior._namespace['frame'];ns['frame']=FunctionType(f.__code__,ns,f.__name__,f.__defaults__,f.__closure__);f=prior.collect;return FunctionType(f.__code__,ns,f.__name__,f.__defaults__,f.__closure__)
def collect(directory,binding,active_actor):return ports(active_actor)(directory,binding)
