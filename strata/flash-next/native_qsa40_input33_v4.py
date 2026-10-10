"""Four repeated exact prefix4 source33 targets vs independent original rows."""
from pathlib import Path
import hashlib
import numpy as np
from original_first_gdn_layer_v1 import OriginalTensorRows
from ple_owned_history_storage_v1 import ngram_rows,PleHashConsts
import native_qsa40_reader_v4 as reader
require=reader.require

def verify(proof,identity,provider=None):
 require(proof['passed']is True and proof['actual_matching_SFD_request_terminal_observed']is True and len(proof['frames'])==4,'Actual complete repeated source33 roster required');require({x['binding']['request']for x in proof['frames']}=={1,2,3,4},'Exact repeated source request roster required')
 if provider is None:provider=OriginalTensorRows(reader.r.HERE/'original-gguf-reference-foundation-plan-v1.json',Path(identity))
 results=[]
 for frame in proof['frames']:
  f=frame['binding'];require(f['gen_ids']==list(reader.qsa.IDS)and f['T']==1 and f['pos']==3 and f['prev']==list(reader.qsa.IDS[1:3])and f['window_tokens']==[reader.qsa.IDS[3]],'Exact actual finalsource window differs');wanted=ngram_rows(reader.qsa.IDS[3],list(reader.qsa.IDS[1:3]),PleHashConsts());raw=reader.consume(frame['fields']['row_ids']['path'],64);require(np.frombuffer(raw,dtype='<u4').tolist()==wanted==f['row_ids'],'Actual source33 rowIDs differ from independent hash');embedding=reader.consume(frame['fields']['embedding']['path'],10240);expected=np.asarray(provider.rows('per_layer_token_embd.weight',wanted),dtype='<f4').reshape(1,2560).tobytes();require(embedding==expected,'Actual source33 IQ4_NL embedding differs from independent original rows');results.append({'request':f['request'],'row_ids':wanted,'embedding_sha256':hashlib.sha256(embedding).hexdigest(),'bitwise_equal':True})
 return {'passed':True,'results':results,'actual_original_staged_input_bitwise':True,'captured_embedding_used_as_math_input':False,'full_model_math_qualified':False}
