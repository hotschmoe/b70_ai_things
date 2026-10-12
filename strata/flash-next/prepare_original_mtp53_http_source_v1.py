"""Pure exact58 HTTP source overlay; explicit original compound schema only."""
import argparse,hashlib,json,shutil,ast
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
PLAN=HERE/'original-mtp53-http-compound-source-plan-v1.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def once(s,a,b):
 if s.count(a)!=1:raise ValueError('Exact HTTP source seam changed '+a[:80])
 return s.replace(a,b,1)
def class_once(s,name,a,b):
 nodes=[n for n in ast.parse(s).body if isinstance(n,ast.ClassDef)and n.name==name]
 if len(nodes)!=1:raise ValueError('Unique actual HTTP class required')
 node=nodes[0];lines=s.splitlines(keepends=True);start=sum(map(len,lines[:node.lineno-1]));end=sum(map(len,lines[:node.end_lineno]));return s[:start]+once(s[start:end],a,b)+s[end:]
def construct(s):
 # Existing imports already have E402 noqa and the parent adds source root.
 s=once(s,'class StrataEngine:', 'from serve import original_mtp53_http_compound as original_compound53\n\nclass StrataEngine:')
 old='    def generate(self, ids, max_new, sampling, cancel, embeddings=None):'
 s=class_once(s,'StrataEngine',old,'''    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        with original_compound53.stream(self,cancel):
            yield from self._generate_original_gate_body(ids,max_new,sampling,cancel,embeddings)

    def _generate_original_gate_body(self, ids, max_new, sampling, cancel, embeddings=None):''')
 s=once(s,'                if alone and left > 1:','                if alone and left > 1 and not original_compound53.mode(self):')
 anchor='        if (embeddings or times >= self.SOLO_AGAIN_MAX or left < self.SOLO_AGAIN_MIN_LEFT or'
 s=once(s,anchor,'        if original_compound53.mode(self):return False\n'+anchor)
 s=once(s,'    def slot_action(self, slot: str, action: str, filename) -> tuple[int, dict]:','    def slot_action(self, slot: str, action: str, filename, compound_sha256=None) -> tuple[int, dict]:')
 anchor='        if not self.slot_save_path or not hasattr(self.engine, "session_file"):'
 s=once(s,anchor,'''        if original_compound53.mode(self.engine):
            return original_compound53.slot_action(self,slot,action,filename,compound_sha256,EngineDied,slot_filename_problem)
'''+anchor)
 s=once(s,'                                                     req.get("filename"))','                                                     req.get("filename"),req.get("compound_sha256"))')
 return s

def prepare(sdk,output):
 plan=json.loads(PLAN.read_bytes())
 for p,h in plan['files'].items():
  if sha(ROOT/p)!=h:raise ValueError('HTTP overlay source changed '+p)
 sdk=Path(sdk);source=sdk/'source/serve/server.py'
 if sha(source.parent/'artifact_identity.py')!=plan['actual58_artifact_identity_sha256']:raise ValueError('Exact frozen artifact identity source required')
 if sha(source)!=plan['actual58_server_sha256']:raise ValueError('Exact frozen actual58 HTTP source required')
 output=Path(output)
 if not output.is_absolute()or output.resolve()!=output or output.exists():raise ValueError('New canonical HTTP overlay root required')
 output.mkdir();serve=output/'serve';inputs={str(p.relative_to(source.parent)):sha(p)for p in source.parent.rglob('*')if p.is_file()};
 for p in source.parent.rglob('*'):
  if p.is_symlink():raise ValueError('Exact nonaliased source package required')
 shutil.copytree(source.parent,serve);body=construct(source.read_text());
 if hashlib.sha256(body.encode('utf8')).hexdigest()!=plan['constructed_server_sha256']:raise ValueError('Exact constructed HTTP source changed')
 identity=serve/'artifact_identity.py';original_identity=identity.read_text();roster='"serve/batch_request_identity.py","serve/server.py","serve/frontend.py","serve/artifact_identity.py","tools/strata_tokenizer.py","tools/gguf_reader.py"';identity.write_text(once(original_identity,roster,roster+',"serve/original_mtp53_http_compound.py"'),encoding='utf-8');(serve/'server.py').write_text(body,encoding='utf-8');(serve/'original_mtp53_http_compound.py').write_bytes((HERE/'original_mtp53_http_compound_v1.py').read_bytes())
 record={'source_plan_sha256':sha(PLAN),'files':{str(p.relative_to(output)):sha(p)for p in serve.rglob('*')if p.is_file()},'original_package_sha256':inputs,'GPU_executed':False,'API_qualified':False,'legacy_path_math_unchanged':True,'native_serving_ELF_required_separately':True};
 if inputs!={str(p.relative_to(source.parent)):sha(p)for p in source.parent.rglob('*')if p.is_file()}:raise ValueError('Source package changed during overlay copy')
 (output/'receipt.json').write_text(json.dumps(record,indent=2)+'\n',encoding='ascii');return record
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--sdk',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(prepare(a.sdk,a.output)))
