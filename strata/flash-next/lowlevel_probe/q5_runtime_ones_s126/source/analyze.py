#!/usr/bin/env python3
import json,sys,statistics
from pathlib import Path
out=Path(sys.argv[1]);log=Path(sys.argv[2]);summary={'status':'FAIL','timing':None}
try:
 rows=[json.loads(line) for line in log.read_text().splitlines() if line.startswith('{')]
 raw=[x for x in rows if x.get('type')=='raw16_gate'];assert [(x['label'],x['word'],x['cells']) for x in raw]==[('extremes-ones',16843009,8),('payload-negative-control',16843010,8),('canonical-a',16843009,320),('canonical-b',16843009,320)]
 assert all(x['INT64_exact'] and x['readonly_guards'] for x in raw)
 snaps=[x for x in rows if x.get('type')=='checked_snapshot'];assert len(snaps)==49
 assert all(x['guards'] and x['tail'] and x['finite'] and x['input_readonly'] and x['weight_guards_checked'] and x['weight_payload_verified'] is False and x['Q8_readonly'] for x in snaps)
 corr=[x for x in rows if x.get('type')=='correctness_pass'];assert len(corr)==1 and corr[0]['shape']=='head'
 timing=[x for x in rows if x.get('type')=='timing_sample'];assert len(timing)==8
 assert rows.index(corr[0])<min(rows.index(x) for x in timing)
 weights=[x for x in rows if x.get('type')=='weight_payload_check'];assert len(weights)==2
 assert [(x['boundary'],x['bytes']) for x in weights]==[('after-correctness-before-timing',437043200),('after-timing-final',437043200)]
 assert all(x['weight_guards_checked'] and x['weight_payload_verified'] and x['shape']=='head' for x in weights)
 assert rows.index(corr[0])<rows.index(weights[0])<min(rows.index(x) for x in snaps if x['phase'].startswith('warm-'))
 assert max(rows.index(x) for x in timing)<rows.index(next(x for x in snaps if x['phase']=='final'))<rows.index(weights[1])
 assert [(x['round'],x['slot'],x['arm']) for x in timing]==[(r,s,a) for r,order in enumerate([['original','candidate','candidate','original'],['candidate','original','original','candidate']]) for s,a in enumerate(order)]
 assert all(x['captured_projection_calls']==20 and not x['quantization_in_timed_graph'] and x['checked_full_bytes'] for x in timing)
 pairs=[x for x in rows if x.get('type')=='paired_mean'];assert len(pairs)==2 and [x['round'] for x in pairs]==[0,1]
 result=json.loads((out/'result.json').read_text());assert result['status']=='PASS' and result['shapes']==1 and result['paired_rounds_per_shape']==2 and not result['serving_qualified']
 assert len(list(out.glob('head/*.bin')))==74 and len(list(out.glob('head/*.dot')))==5
 original=[x['original_complete_us_per_call'] for x in pairs];candidate=[x['candidate_complete_us_per_call'] for x in pairs]
 summary={'status':'PASS_INTERNAL_ORACLE_COUNTS_ONLY','raw16_cells':sum(x['cells'] for x in raw),'snapshots':49,'whole_weight_payload_checks':2,'whole_weight_bytes_verified':874086400,'paired_rounds':2,'samples':8,'timing':{'original_process_nested_median_us':statistics.median(original),'candidate_process_nested_median_us':statistics.median(candidate),'per_round_percent':[100*(b/a-1) for a,b in zip(original,candidate)]},'DOT_binding_required_separately':True,'serving_qualified':False,'population_inference':False}
except Exception as e:
 summary['error']=str(e);(out/'analysis.json').write_text(json.dumps(summary,indent=2)+'\n');raise
(out/'analysis.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
