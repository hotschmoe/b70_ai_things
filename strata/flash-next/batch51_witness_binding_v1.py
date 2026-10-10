"""Extra READY full-byte observation joined to the unchanged three boundary gates."""
import copy,math
from serial37_canonical_json_v3 import canonical
from operation_pack_hash_witness_v2 import require
from pack_witness_binding47_v1 import binding as pack_original
from sdk_witness_binding49_v1 import binding as sdk_original

def binding(proof,epoch,owner,leaf,terminal,kind,ready,ack_epoch):
 rows=proof['boundaries'];labels=['admission_start','ready_complete_byte_recheck','predevice_complete_byte_recheck','postoperation_complete_byte_recheck'] if kind=='pack' else ['entry','ready','predevice','postoperation']
 require(len(rows)==4 and [r['label'] for r in rows]==labels,'Exact four independent child byte boundaries required');require(canonical(rows[1])==canonical(ready) and canonical(rows[1]['rows'])==canonical(epoch.initial),'Actual READY/current full-byte witness differs')
 if kind=='sdk':require(canonical(rows[1]['receipt_inputs'])==canonical(epoch.current_receipts),'READY current SDK receipt union differs')
 require(all(type(row[k])in (int,float) and math.isfinite(row[k]) for row in rows for k in ('started_epoch','finished_epoch')) and rows[0]['finished_epoch']<=rows[1]['started_epoch']<=rows[1]['finished_epoch']<=ack_epoch<=rows[2]['started_epoch'],'Actual ACK-to-predevice complete byte chronology differs')
 stripped=copy.deepcopy(proof);stripped['boundaries']=[rows[0],rows[2],rows[3]];result=(pack_original if kind=='pack' else sdk_original)(stripped,epoch,owner,leaf,terminal);result['complete_byte_boundaries']=4;result['ready_boundary_rejoined']=True;return result
