"""Immutable source40 compile-name correction over frozen observer V1."""
import difflib,hashlib,json
from pathlib import Path
import prepare_native_layer3_qsa_observer_v1 as old
ROOT=old.ROOT;HERE=old.HERE
BASE=HERE/'native-layer3-qsa-observer-engine-build-plan-v1.json'
BASE_SHA='72ad297b5321b66237abfd42aca6dc4199fcda38d63bf080d211a2ddeb6abbca'
PLAN=HERE/'native-layer3-qsa-observer-engine-build-plan-v2.json'
PLAN_SHA='d87740b5cbd5310895b057541ae5537034b993e38bb20e6248350a7452dbc2e6'
PATCH=HERE/'patches/0040-default-off-native-layer3-qsa-target-observer-v2.patch'
HEADER=old.HEADER;H=old.H;sha=old.sha

def reconstruct():
 if sha(BASE)!=BASE_SHA:raise ValueError('Frozen failed source40V1 plan changed')
 base,previous=old.reconstruct();new=dict(previous);path='sycl/src/core/verify.cpp';s=new[path];start=s.index('    if(qsa3_target::settings().enabled) {');end=s.index('    if (std::getenv("STRATA_VERIFY_DEBUG")',start);block=s[start:end]
 for name in ('native_qsa_enabled','native_rope_enabled','native_qsa_indexer_enabled'):
  if block.count('!'+name+'()')!=1:raise ValueError('Exact observer predicate count changed '+name)
  block=block.replace('!'+name+'()','!kernels::'+name+'()')
 new[path]=s[:start]+block+s[end:];return base,new

def patch_bytes():
 base,new=reconstruct();return ''.join(''.join(difflib.unified_diff(base.get(n,'').splitlines(True),new[n].splitlines(True),fromfile='a/'+n if n in base else '/dev/null',tofile='b/'+n))for n in sorted(new)if base.get(n)!=new[n]).encode('ascii')

def build_plan():
 if PATCH.read_bytes()!=patch_bytes() or sha(PLAN)!=PLAN_SHA:raise ValueError('Actual corrected frozen patch/plan changed')
 base=json.loads(BASE.read_bytes());plan=json.loads(PLAN.read_bytes());_,new=reconstruct();expected={n:hashlib.sha256(t.encode()).hexdigest()for n,t in new.items()}
 if plan['expected_patched_source_sha256']!=expected or len(expected)!=67 or len(plan['added_header_payloads'])!=31 or len(plan['patches'])!=40:raise ValueError('Actual complete corrected reconstruction differs')
 if plan['patches'][:-1]!=base['patches'][:-1] or plan['patches'][-1]!={'path':str(PATCH.relative_to(ROOT)),'sha256':sha(PATCH)}:raise ValueError('Original39/corrected40 ordered patches changed')
 changed={k for k in set(plan)|set(base) if plan.get(k)!=base.get(k)}
 if changed!={'expected_patched_source_sha256','patches','derived_from_plan','derived_from_plan_sha256','namespace_compile_correction'}:raise ValueError('Correction changed another original recipe field')
 if plan['derived_from_plan']!=str(BASE.relative_to(ROOT)) or plan['derived_from_plan_sha256']!=BASE_SHA or plan['added_header_payloads']!=base['added_header_payloads']:raise ValueError('Original input header provenance changed')
 if plan['namespace_compile_correction']['model_arithmetic_changed'] is not False or plan['namespace_compile_correction']['actual_compilation_qualified'] is not False:raise ValueError('Source correction cannot grant arithmetic/compilation qualification')
 return plan
