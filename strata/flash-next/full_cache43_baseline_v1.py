"""Actual API43 proof captured while174; current175 never becomes an old gate."""
import ast,copy
from pathlib import Path
import c1_source43_api_controller_v3 as original
import source43_api_prerequisite_bytes_v1 as history
import registry_source43_cache175_association_v1 as registry
from serial37_canonical_json_v3 import canonical
require=original.require
RECORD=Path('/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/source43-api-for-cache-baseline-v1.json')

def capture(root):
    from validate_source43_api_baseline_v3 import finalized_binding
    require(original.registry_binding()['actual_models']==174,'Genuine completed API43 prerequisite must be admitted while174 current');proof=finalized_binding(root);return history.capture(root,proof,'genuine43mode1_API8_before_cache')

def validate_metadata(root):
    # Explicit named consumer route: preserve all other current original gates.
    node=copy.deepcopy(next(n for n in ast.parse(Path(original.__file__).read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='validate_prepared'));calls=[n for n in ast.walk(node)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='registry_binding'];require(len(calls)==1,'Exact original current registry call required');calls[0].func.id='cache_registry';namespace={**vars(original),'cache_registry':lambda:{**registry.association(original.REPO/'evals/configs/models.yaml'),'new_alias_registered':True}};exec(compile(ast.fix_missing_locations(ast.Module([node],type_ignores=[])),'[explicit43cache-current175-metadata]','exec'),namespace);return namespace['validate_prepared'](Path(root))

def finalized_binding(root,record=None):
    if record is None:record=original.read(RECORD)
    require(record['role']=='genuine43mode1_API8_before_cache','Actual named API43 prerequisite required');history.rejoin(record);require(Path(record['root'])==Path(root).resolve(),'Actual baseline root differs');metadata=validate_metadata(root);require(record['original_public_admission']['source_generation']==43 and record['original_public_admission']['materialized_mode']==1 and record['original_public_admission']['actual_HTTP_overlap_passed']is True,'Actual completed native43 API baseline required');return metadata,{'original_completed_API43_binding':record['original_public_admission'],'historical_exact_raw_byte_roster':record['byte_roster'],'old174_current_registry_gate_passed':False,'current175_registry':registry.association(original.REPO/'evals/configs/models.yaml'),'new_leaf_health_authority_transferred':False,'full_model_math_authority_transferred':False}

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--api-root',type=Path,required=True);a=p.parse_args();require(not RECORD.exists(),'New immutable API43 cache prerequisite receipt required');original.write(RECORD,capture(a.api_root))
