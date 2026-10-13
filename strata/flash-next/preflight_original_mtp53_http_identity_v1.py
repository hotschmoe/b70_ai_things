"""Actual modified HTTP identity load/validation only; root NoDri execution required."""
import argparse,json,hashlib,sys
from pathlib import Path
def main(a):
 if Path('/dev/dri').exists():raise ValueError('HTTP identity preflight must expose no GPU')
 path=Path(a.config);cfg=json.loads(path.read_bytes());manifest=Path(cfg['artifact_identity_manifest']);digest=hashlib.sha256(manifest.read_bytes()).hexdigest()
 if digest!=a.manifest_sha256:raise ValueError('Pinned actual HTTP manifest required')
 sys.path[:0]=['/src','/src/tools']
 from serve import server,artifact_identity
 identity=artifact_identity.load_identity(cfg,Path(cfg['tokenizer']))
 env=server.child_env(cfg);args=server.engine_args(cfg)
 artifact_identity.validate_runtime(identity,cfg['exe'],args,env)
 if identity['sha256']!=digest or cfg['model_name']!='hotschmoe-dd' or len(cfg['aliases'])!=1:raise ValueError('Actual load identity/primary alias required')
 if args!=cfg['args']:raise ValueError('HTTP resolution changed native arguments')
 if set(identity['manifest']['runtime']['python_sources'])!=set(artifact_identity.PYTHON_SOURCES):raise ValueError('Actual complete modified HTTP source contract required')
 print(json.dumps({'passed':True,'manifest_sha256':digest,'actual_load_identity_executed':True,'actual_validate_runtime_executed':True,'actual_engine_args_and_child_env_executed':True,'qualified_interpreter':sys.executable,'primary_model_name':cfg['model_name'],'research_aliases':cfg['aliases'],'original_first_GGUF_full_hash_executed':True,'export_vs_original_GGUF_metadata_checked':True,'native_engine_started':False,'GPU_exposed':False,'API_qualified':False,'cache_qualified':False},ensure_ascii=True))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--manifest-sha256',required=True);main(p.parse_args())
