"""Fresh tool overlay/build metadata from full fixed-budget source byte closure."""
import argparse,hashlib,json,types
from pathlib import Path
import prepare_pti_explicit_finalize_v1 as change
import pti_unitrace_build_recipe_v1 as original
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];MANIFEST=HERE/'pti-unitrace-source-manifest-v1.json';BRIDGE=HERE/'pti-unitrace-clock-bridge-source-plan-v1.json';BUDGET=HERE/'pti-unitrace-clock-budget-source-plan-v1.json';PATCH=HERE/'pti-explicit-model-finalizer-v1.patch'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def parent_binding():
    require=change.require;require(sha(MANIFEST)=='7593e5381a7373cebbe861b0ffb464afaa5374db0c122972d638c39849b82a11','Exact original source manifest required');manifest=json.loads(MANIFEST.read_bytes());overrides=dict(json.loads(BRIDGE.read_bytes())['expected_reconstructed_source_sha256']);overrides.update(json.loads(BUDGET.read_bytes())['expected_reconstructed_source_sha256']);expected={};links={}
    for name,row in manifest['tracked_files'].items():
        path=change.BASE/name
        if 'symlink_target'in row:require(path.is_symlink()and path.readlink().as_posix()==row['symlink_target'],'Exact source symlink changed '+name);links[name]=row['symlink_target']
        else:
            want=overrides.get(name,row['sha256']);require(path.is_file()and not path.is_symlink()and sha(path)==want,'Exact full fixed-budget source changed '+name);expected[name]=want
    for name,want in overrides.items():require(sha(change.BASE/name)==want,'Exact fixed-budget added/modified source changed '+name);expected[name]=want
    return expected,links

def proposal():
    expected,links=parent_binding();_,new=change.changes();expected.update({name:hashlib.sha256(body.encode()).hexdigest()for name,body in new.items()});return {'schema':1,'status':'SOURCE ONLY fresh model-finalizer tool build; no old ELF or runtime authority','source_commit':original.COMMIT,'source_parent':str(change.BASE),'source_parent_original_manifest_sha256':sha(MANIFEST),'source_parent_bridge_plan_sha256':sha(BRIDGE),'source_parent_budget_plan_sha256':sha(BUDGET),'patch':{'path':str(PATCH.relative_to(ROOT)),'sha256':sha(PATCH)},'expected_patched_source_sha256':{name:hashlib.sha256(body.encode()).hexdigest()for name,body in new.items()},'expected_full_source_sha256':expected,'expected_source_symlinks':links,'source_file_count':len(expected)+len(links),'original_tracked_count':893,'new_header_count':2,'image':original.IMAGE,'compiler':{'path':original.ICXX,'sha256':original.ICXX_SHA},'level_zero_revision':original.LEVEL_ZERO,'compute_runtime_revision':original.COMPUTE_RUNTIME,'build_targets':['unitrace','unitrace_tool'],'metadata_cap_bytes':512<<20,'row_cap_bytes':4096,'defaultOFF_role':'UNITRACE_MODEL_FINALIZE_V1 absent/0','explicit_export':'b70_pti_model_finalize_v1','full_fresh_build_required':True,'old_budget_ELF_build_receipt_transferred':False,'actual_build_executed':False,'runtime_ready':False}

def validate_materialized(source,value):
    source=Path(source).resolve();change.require(source!=change.BASE and source.is_dir()and not source.is_symlink(),'New independent finalizer source tree required')
    for name,want in value['expected_full_source_sha256'].items():
        path=source/name;change.require(path.is_file()and not path.is_symlink()and sha(path)==want,'Fresh exact finalizer materialized source differs '+name)
    for name,target in value['expected_source_symlinks'].items():change.require((source/name).is_symlink()and(source/name).readlink().as_posix()==target,'Fresh source symbolic-link roster differs '+name)
    return True

def recipe(materialized_source,output,proposal_sha256):
    # Private immutable original recipe code with explicit new source/label only.
    namespace={**vars(original),'SOURCE':Path(materialized_source),'MANIFEST_SHA':proposal_sha256};function=types.FunctionType(original.recipe.__code__,namespace,'fresh_finalizer_recipe',original.recipe.__defaults__);return function(Path(output))
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--materialized-source',type=Path,required=True);parser.add_argument('--build-output',type=Path,required=True);args=parser.parse_args();value=proposal();validate_materialized(args.materialized_source,value);print(json.dumps({'proposal_sha256':sha(HERE/'pti-finalizer-tool-build-proposal-v1.json'),'command':recipe(args.materialized_source,args.build_output,sha(HERE/'pti-finalizer-tool-build-proposal-v1.json')),'actual_build_executed':False},indent=2))
if __name__=='__main__':main()
