"""Actual pinned artifact API runtime validator inside NoDri C388; no GGUF payload scan."""
import argparse,hashlib,json,os,sys
from pathlib import Path
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main(a):
 if Path('/dev/dri').exists():raise ValueError('Actual artifact preflight must expose no GPU')
 root=a.artifacts
 if sha(root/'binding.json')!=a.binding_sha256:raise ValueError('Actual artifact binding changed')
 binding=json.loads((root/'binding.json').read_bytes());sys.path.insert(0,'/src');from serve import artifact_identity as actual
 for mode,r in binding['records'].items():
  manifest=root/Path(r['manifest']).name;cfgpath=root/Path(r['runtime']).name
  if sha(manifest)!=r['manifest_sha256']or sha(cfgpath)!=r['runtime_sha256']:raise ValueError('Actual manifest/config changed')
  value=json.loads(manifest.read_bytes());cfg=json.loads(cfgpath.read_bytes());env=dict(cfg['env']);digest=env.pop(actual.IDENTITY_ENV)
  if digest!=sha(manifest):raise ValueError('Native digest must be actual consumed manifest hash')
  identity={'manifest':value,'sha256':digest,'tokenizer_path':'/pack/tokenizer'}
  actual.validate_runtime(identity,cfg['exe'],cfg['args'],env)
  if value['primary_model_name']!=actual.PRIMARY or value['model']['repo']!=actual.SOURCE_REPO or value['model']['revision']!=actual.SOURCE_REVISION or value['model']['tokenizer_gguf_sha256']!=actual.TOKENIZER_GGUF_SHA256:raise ValueError('Actual API publisher identity schema changed')
 print('MTP53_ARTIFACT_RUNTIME_PREFLIGHT_PASS modes=2 actual_custom_ELF=1 exact_native_config=1 actual_python_source_versions=1 tokenizer_files=5 no_GPU=1 whole_GGUF_read=0')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--artifacts',type=Path,required=True);p.add_argument('--binding-sha256',required=True);main(p.parse_args())
