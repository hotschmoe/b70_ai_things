"""Actual serial protocol witness/call binding/shell parse, before model launch."""
import argparse,hashlib,inspect,json,subprocess,tempfile
from unittest.mock import patch
from pathlib import Path
import run_source62_cached_window_diagnostic_v1 as actor
import validate_source62_cached_window_diagnostic_v1 as reader
import source62_cached_window_profile_contract_v1 as protocol

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind_call(fn,args=(),kwargs=None):
 inspect.signature(fn).bind(*args,**(kwargs or {}));return dict(callable_module=fn.__module__,callable_qualname=fn.__qualname__,actual_signature=str(inspect.signature(fn)),arity_passed=True)
def shell_syntax(script):
 require=actor.require;require(type(script)is str and '\x00'not in script,'Actual shell text required');r=subprocess.run(['bash','-n'],input=script,text=True,capture_output=True,timeout=10);require(r.returncode==0,'Actual native shell syntax failed: '+r.stderr[:1000]);return dict(shell_parse_passed=True,script_sha256=hashlib.sha256(script.encode()).hexdigest(),shell_execution_performed=False)
def protected_constructor_binding(command,recipe):
 # Execute the actual RAW observer constructor through its real superclass call.
 # Stop only at the original underlying constructor, before intent/Popen/Docker.
 cls=actor.NS['Protocol'];base=cls.__bases__[0];original=base.__init__;called=[]
 class BeforeLaunch(Exception):pass
 def stop(self,*args,**kwargs):
  inspect.signature(original).bind(self,*args,**kwargs);called.append(dict(original_module=original.__module__,actual_signature=str(inspect.signature(original)),forwarded_kwargs=sorted(kwargs)));raise BeforeLaunch()
 with tempfile.TemporaryDirectory()as t:
  directory=Path(t);(directory/'recipe.json').write_text(json.dumps(recipe))
  with patch.object(base,'__init__',stop):
   try:cls(command,directory,lambda:None,active=lambda:None)
   except BeforeLaunch:pass
 actor.require(len(called)==1,'Actual superclass constructor forwarding failed');return called[0]

def check(construction,construction_sha):
 require=actor.require;require(sha(construction)==construction_sha,'Exact actual constructor receipt required');value=json.loads(Path(construction).read_bytes());require(value['pure_command_construction_passed']is True and value['GPU_launch_performed']is False,'Actual CPU constructor only');prior=json.loads((actor.HERE/'source62-diagnostic-prior-identity-v1.json').read_bytes());root=Path(prior['root']);require(sha(root/'report.json')==prior['report_sha256'] and sha(prior['public'])==prior['public_sha256'],'Actual authenticated serial witness changed');report=json.loads((root/'report.json').read_bytes());phase='source62-cache1536-copy0-untraced-companion';recipe=json.loads((root/phase/'recipe.json').read_bytes());require(sha(root/phase/'recipe.json')==report['artifacts'][phase+'/recipe.json'] and sha(root/phase/'requests.json')==report['artifacts'][phase+'/requests.json'],'Actual request/recipe witness bytes changed');rows=json.loads((root/phase/'requests.json').read_bytes());views=protocol.requests(rows,recipe['requests']);require([x['actual_reused']for x in views]==[0,1536] and [x['new_prompt_rows']for x in views]==[2048,512],'Actual observed native serial/PIN/RID/DONE witness differs')
 calls=[bind_call(protocol.requests,(rows,recipe['requests'])),bind_call(reader.finalized_binding,(root,)),bind_call(actor.NS['Protocol'].__bases__[0].request,(None,'request-1',[1]*2048),dict(max_new=8,fresh=1,pin=1536))]
 commands=[]
 for x in value['commands']:
  command=x['native_command'];require(isinstance(command[-1],str),'Actual generated native shell required');commands.append(dict(shell=shell_syntax(command[-1]),actual_constructor=protected_constructor_binding(command,x['recipe'])))
 require(sha(construction)==construction_sha,'Constructor changed during CPU preflight')
 return dict(passed=True,scope='Exact current serial cache protocol and Python call signatures plus shell syntax only; future observer runtime consumes/startup allocations remain unproven',actual_native_serial_witness=True,original_native_protocol_parsed=True,calls=calls,shells=commands,GPU_model_or_compiler_started=False)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--construction',type=Path,required=True);p.add_argument('--construction-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();actor.require(not a.output.exists(),'New preflight receipt required');a.output.write_text(json.dumps(check(a.construction,a.construction_sha256),indent=2)+'\n',encoding='ascii')
