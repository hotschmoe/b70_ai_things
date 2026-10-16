from pathlib import Path
import json,sys
p=Path(sys.argv[1]);rows=[json.loads(x)for x in(p/'leaf.stdout').read_text().splitlines()if x.startswith('{')]
strict=[r for r in rows if r.get('type')=='strict'];snap=[r for r in rows if r.get('type')=='snapshot'];times=[r for r in rows if r.get('type')=='timing'];final=[r for r in rows if r.get('status')]
assert len(strict)==9 and len(snap)==18 and len(times)==8 and len(final)==1
assert final[0]['status']=='PASS_PRIVATE_GDN_LAYER_BATCH'and final[0]['complete_state_bytes']==235339776 and final[0]['snapshot_receipts']==18 and final[0]['fullsnapshotfiles']==0
assert {(x['acceptedA'],x['acceptedB'],x['phase'])for x in strict}=={(n,4-n,p)for n in[1,2,3]for p in[0,1,2]}
assert all(x['layers']==36 and x['owners']==2 and x['rows']==3 and x['all_state_bytes']==235339776 and all(x[k]is True for k in['guards','finite','readonly','output_sentinel','snapshot_continuation'])for x in strict)
assert {(x['pair'],x['phase'],x['kind'])for x in snap}=={(n,p,k)for n in[1,2,3]for p in[0,1,2]for k in['eager','continuation-eager']}
assert all(x['complete_bytes_compared']==235339776 and x['digest_not_numeric_comparator']is True for x in snap)
assert [x['arm']for x in times]==[0,1,1,0,1,0,0,1]
assert all(x['completed_host_chain_us']>0 and x['conv_calls']==x['state_calls']==(2 if x['arm']else 72)for x in times)
assert not list((p/'leaf').iterdir()),'no duplicate full-state snapshot files'
summary={'status':'PASS_PRIVATE_STRICT_ONLY','strict':final[0],'phase_records':strict,'accepted_state_digest_receipts':snap,'digest_scope':'FNV1a64 descriptive receipts only; complete memcmp+finite+guards performed in process, captured hashes omitted as redundant after exact eager-byte equality','timings':times,'timing_scope':'captured completed HOST GDN-only chain,72conv+72state vs2conv+2state; all setup/check/readback outside timedwindow; no serving forecast','speed_or_serving_qualified':False}
(p/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='ascii');print(json.dumps({'status':summary['status'],'phases':len(strict),'digest_receipts':len(snap),'timings':len(times)}))
