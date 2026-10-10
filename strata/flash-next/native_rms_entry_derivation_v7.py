"""Byte-exact function-name-only adaptation; no leaf math change."""
import hashlib,json
from pathlib import Path
PROPOSAL_SHA='31d294c1412e79444a45d1a8dd90ea856712fc1a9a2799b562e6964618ea6602'
WRAPPER_PLAN_SHA='7ee7968d44c030d8c99ccd72acf3f8c460736a8e446859c86ab41299be1b4354'
FUNCTION=b'int main(int argc,char** argv) try {'
CALLABLE=b'int rms37_frozen_proposal_main(int argc,char** argv) try {'
OLD_INCLUDE=b'#define main rms37_frozen_proposal_main\n#include "native_rms_rsqrt37_gpu_v1.cpp"\n#undef main\n'
NEW_INCLUDE=b'#include "native_rms_rsqrt37_callable_v7.cpp"\n'
def sha(raw):return hashlib.sha256(raw).hexdigest()
def require(ok,msg):
 if not ok:raise ValueError(msg)
def prove(proposal,callable_source,wrapper,entry):
 require(proposal.count(FUNCTION)==1 and callable_source==proposal.replace(FUNCTION,CALLABLE),'Callable copy differs beyond sole function identifier')
 require(wrapper.count(OLD_INCLUDE)==1 and entry==wrapper.replace(OLD_INCLUDE,NEW_INCLUDE),'Metadata wrapper differs beyond exact include adaptation')
 return {'original_proposal_sha256':sha(proposal),'callable_sha256':sha(callable_source),'original_wrapper_sha256':sha(wrapper),'entry_sha256':sha(entry),'leaf_function_identifier_only':True,'metadata_wrapper_include_only':True,'leaf_math_changed':False}
def admit(here):
 here=Path(here);plans=[]
 for name,want in [('native-rms-rsqrt37-source-plan-v1.json',PROPOSAL_SHA),('native-rms-rsqrt37-owned-source-plan-v1.json',WRAPPER_PLAN_SHA)]:
  raw=(here/name).read_bytes();require(sha(raw)==want,'Original frozen derivation authority changed');plans.append(json.loads(raw))
 names=['native_rms_rsqrt37_gpu_v1.cpp','native_rms_rsqrt37_callable_v7.cpp','native_rms_rsqrt37_owned_entry_v1.cpp','native_rms_rsqrt37_owned_entry_v7.cpp'];raw=[(here/n).read_bytes() for n in names]
 for plan,i in zip(plans,(0,2)):require(plan['files']['strata/flash-next/'+names[i]]==sha(raw[i]),'Original frozen leaf/wrapper bytes changed')
 return prove(*raw)
