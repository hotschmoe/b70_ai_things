"""Root-owned metadata removal retains its original receipt through absence."""
import json
from pathlib import Path

def retire(name,owned,output,ledger,inspect,run,pause):
 def require(ok,message):
  if not ok:raise ValueError(message)
 def save():Path(output,'metadata-ownership-ledger.json').write_text(json.dumps(ledger,indent=2,ensure_ascii=True)+'\n',encoding='ascii')
 removed=False;terminal=None
 while True:
  try:
   if not removed:
    obj=inspect(name);require(obj is not None,'Metadata creation ownership unobserved; exclusion retained');owned(obj);ledger['actual_owned_container_observed']=True;save()
    if obj['State']['Running']:
     row=run(['docker','stop','--time','10',name],capture_output=True,text=True,timeout=30);require(type(row.returncode)is int and row.returncode==0,'Exact owned metadata stop failed');ledger['forced_owned_stop']=True;save();continue
    terminal=obj;ledger['terminal_inspection']=obj;save();command=['docker','rm',name];row=run(command,capture_output=True,text=True,timeout=30);ledger['original_successful_removal_receipt']={'command':command,'return_code':row.returncode,'original_container_id':obj['Id']};require(type(row.returncode)is int and row.returncode==0,'Exact owned metadata removal failed');removed=True;save()
   require(inspect(name)is None,'Original successful removal still lacks authoritative current absence')
   ledger['owned_terminal_and_removed']=True;save();return terminal
  except BaseException as error:
   ledger['retirement_failure_count']=ledger.get('retirement_failure_count',0)+1;failures=ledger.setdefault('retirement_failures',[])
   if len(failures)<64:failures.append(type(error).__name__+': '+str(error))
   ledger['last_retirement_failure']=type(error).__name__+': '+str(error);ledger['retirement_failure_detail_truncated']=ledger['retirement_failure_count']>64;save();pause(.1)
