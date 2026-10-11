"""Eight genuine43 native/API joins including independent text and live owner."""
from pathlib import Path
import source43_api_evidence_v1 as original
import source43_api_live_owner_decode_v1 as joins
from serial37_canonical_json_v3 import canonical
require=original.require;epoch=original.epoch

def bind(root):
 root=Path(root);trace=[joins.json.loads(line)for line in(root/'engine-token-trace.jsonl').read_text().splitlines()];require(all(type(r['engine_pid'])is int and r['engine_pid']>0 and ('call'not in r or type(r['call'])is int and r['call']>0)for r in trace),'Exact typed original trace owner/calls required');base=original.bind(root);m=joins.c.read(root/'prepared.json');launch=joins.c.read(root/'launch.json');owner=joins.c.read(root/'native-owner.json');actual=joins.owner_binding(owner,m,launch);require(actual['namespace_pid']==base['engine_pid'],'Actual native PID and live namespace owner differ');decoded=joins.decoded_binding(root,m,joins.c.read(root/'report.json')['producer_pid']if(root/'report.json').exists()else joins.os.getpid(),joins.c.sha(root/'source-plan.snapshot.json'));require(epoch(owner['epoch'])<=epoch(base['native_closed_epoch'])and epoch(joins.c.read(root/'native-reply-decode/receipt.json')['finished_epoch'])<=epoch(base['native_closed_epoch']),'Actual owner/decode must precede original native close');
 if(root/'report.json').exists():
  report=joins.c.read(root/'report.json');require(canonical(report['native_owner'])==canonical(owner)and canonical(report['independent_reply_decode'])==canonical(decoded),'Original saved owner/decode proof fields differ')
 return {**base,'actual_native_host_owner':actual,'independent_native_ID_text_decode':decoded}
