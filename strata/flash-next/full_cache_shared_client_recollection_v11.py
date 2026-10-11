"""Original SSE/client packet recollection; independent ID/text decode separate."""
import json,math
from pathlib import Path
from serial37_canonical_json_v3 import read_unique,canonical
from full_cache_shared_history_v2 import require

def typed(value):return json.dumps(canonical(value),sort_keys=True,separators=(',',':'),ensure_ascii=True)
def recollect(directory,saved,messages,bounds,policy,alias):
 directory=Path(directory);require(read_unique(directory/'client.json')==saved and type(saved['rows'])is list and len(saved['rows'])==len(messages)==len(bounds) and saved['requested_streams']==len(messages),'Original exact client/row roster required')
 models=read_unique(directory/'models.json');require([r['id']for r in models['data']]==['hotschmoe-dd',alias],'Original percohort model identity differs');texts=[]
 for index,row in enumerate(saved['rows']):
  rid=f'full-cache-{directory.parent.name}-{directory.name}-{len(messages)}-{index}';require(row['request_id']==rid and row['client_index']==index and read_unique(directory/(rid+'.json'))==row,'Original independently saved client row differs');body={'model':'hotschmoe-dd','messages':messages[index],'stream':True,'temperature':0,'max_tokens':bounds[index],'frequency_penalty':0,'presence_penalty':0,'user':rid,'chat_template_kwargs':{'enable_thinking':False},'reasoning_budget_tokens':0};body.update(policy);require(typed(row['request'])==typed(body),'Original exact serial/concurrent request body differs')
  require(type(row['sent_epoch']) in (int,float) and type(row['finished_epoch']) in (int,float) and math.isfinite(row['sent_epoch']) and math.isfinite(row['finished_epoch']) and row['sent_epoch']<=row['finished_epoch'] and row['error'] is None and row['http_status']==200,'Original actual client transport/epoch invalid');text='';done=False;previous=row['sent_epoch']
  for event in row['events']:
   require(type(event['epoch']) in (int,float) and math.isfinite(event['epoch']) and previous<=event['epoch']<=row['finished_epoch'],'Original SSE epoch ordering differs');previous=event['epoch'];require(not done,'Original SSE content after terminal')
   if event['data']=='[DONE]':done=True;continue
   value=json.loads(event['data']);require(not value.get('error') and len(value.get('choices',[]))==1,'Original SSE error/choice roster invalid');piece=value['choices'][0].get('delta',{}).get('content');require(piece is None or type(piece)is str,'Original actual SSE content not text');text+=piece or ''
  require(type(row['cancel_requested'])is bool and type(row['done_received'])is bool and row['done_received']==done and (done or row['cancel_requested']),'Original native/client completion or intentional cancellation missing');texts.append(text)
 return {'original_request_and_SSE_rows_recollected':True,'HTTP_texts':texts,'independent_native_ID_to_text_decode_still_required':True,'API_coherence_or_full_cache_qualified':False}
