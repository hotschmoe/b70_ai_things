"""Future profiling-only normal history route; current admission is unchanged."""
from pathlib import Path
import source43_normal_historical_association_v1 as history
from serial37_canonical_json_v3 import canonical

def prerequisite(root):
 history.require(Path(root).resolve()==history.NORMAL,'Only named actual normal43 history supported');return history.binding()

def admission(plan,original_admission):
 # Original caller still runs its full current SDK/config/masked/model predicates.
 metadata,config=original_admission(plan);associated=history.binding();matched=history.match_current_contract(associated,plan['source43'],config);history.require(matched['current_SDK_bytes_verified_by_this_function']is False and matched['fresh_per_arm_identity_and_runtime_gates_still_required']is True,'Field association cannot become current byte authority');return metadata,config
