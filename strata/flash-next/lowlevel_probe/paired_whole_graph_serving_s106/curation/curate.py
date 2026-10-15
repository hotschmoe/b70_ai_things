from pathlib import Path
import argparse, hashlib, json
REPO=Path('/mnt/vm_8tb/github/b70_ai_things')
R=Path('/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014')
PREP=R/'paired-whole-graph-s106-curation-prep'
PLAN=PREP/'curation-plan.json'
EXPECTED='b73cd131356f4a02f9c8b58355fa552f73c1c6148dc6b05b381bf01ddada422a'
S106='strata/flash-next/lowlevel_probe/paired_whole_graph_serving_s106'
S104='strata/flash-next/lowlevel_probe/q6_aligned_load_s104/post-review'
p=argparse.ArgumentParser();p.add_argument('--execute-after-root-go',action='store_true');args=p.parse_args()
def sha(b):return hashlib.sha256(b).hexdigest()
assert sha(PLAN.read_bytes())==EXPECTED
plan=json.loads(PLAN.read_text())
entries=[]
for group in plan['groups']:
 for f in group['files']:
  entries.append({'policy':'EXACT_COPY','source':str(Path(group['source_root'])/f['source']),'target':str(Path(group['target_root'])/f['target']),'group':group['label']})
for x in plan['external_or_derived_records']:
 e=dict(x);e['policy']='DERIVED' if x['policy'].startswith('DERIVE_') else 'EXTERNAL_ONLY';entries.append(e)
# Root-approved supplemental records, following the plan's same compact recipes.
api=R/'paired-whole-graph-s106-api-prep'
for mode in ('mode0','mode1'):
 entries.append({'policy':'DERIVED','source':str(api/mode/'native-process-identity.json'),'target':f'{S106}/api/runtime/{mode}/native-process-identity-summary.json','reason':'Root-approved same compact native identity recipe; full map external.'})
 for phase in ('warm','profile'):
  for stream in (0,1):
   entries.append({'policy':'DERIVED','source':str(api/mode/'coding'/phase/f'coding-{stream}.json'),'target':f'{S106}/api/runtime/{mode}/coding/{phase}/coding-{stream}-summary.json','reason':'Root-approved same compact coding request recipe; full SSE external.'})
for source,target in [
 (R/'s106-next-vocab-source-audit/note.md',f'{S106}/source-vocab-audit/note.md'),
 (R/'s106-next-vocab-source-audit/pins.json',f'{S106}/source-vocab-audit/pins.json'),
 (api/'root-source-decision.json',f'{S106}/api/root-source-decision.json'),
 (R/'paired-whole-graph-s106-prep/root-clean-decision.json',f'{S106}/clean/root-clean-decision.json'),
 (R/'paired-whole-graph-s106-prep/latency-summary.json',f'{S106}/clean/latency-summary.json')]:
 entries.append({'policy':'EXACT_COPY','source':str(source),'target':target,'group':'Root-approved supplement'})
# Actual one-process summary CSV selected by the finalized closure, not a wildcard guess.
for mode in ('mode0','mode1'):
 c=api/mode/'closure.json'
 if c.is_file():
  d=json.loads(c.read_text());source=Path(d['summary']);assert source.parent==api/mode/'trace'
  entries.append({'policy':'EXACT_COPY','source':str(source),'target':f'{S106}/api/runtime/{mode}/trace/{source.name}','group':'Actual process-specific host summary CSV'})
seen=set()
for e in entries:
 s=Path(e['source']).resolve();assert s.is_relative_to(R)
 if 'target' in e:
  t=Path(e['target']);assert not t.is_absolute() and '..' not in t.parts
  assert str(t).startswith(S106+'/') or str(t).startswith(S104+'/')
  assert str(t) not in seen,str(t);seen.add(str(t))
if not args.execute_after_root_go:
 missing=[e['source'] for e in entries if not Path(e['source']).is_file()]
 print(json.dumps({'status':'PREPARED_NO_COPIES_WAIT_ROOT_GO','entries':len(entries),'targets':len(seen),'missing_now':missing}));raise SystemExit(0)
assert (api/'comparison.json').is_file(),'Final API comparison required before curation'
# First attempt may have written only this script's target prefix before external-index validation.
# Existing targets are accepted solely after byte-exact verification; never overwrite.
for owned_root in (REPO/S106,REPO/S104):
 if owned_root.exists():
  assert not (owned_root/'record-copies.json').exists(),'Final index already exists; do not duplicate curation'
  for existing in owned_root.rglob('*'):
   if existing.is_file():assert str(existing.relative_to(REPO)) in seen,('Unexpected existing target',str(existing))
index=[]
# Validate all required inputs before writing anything. Final root decision is added separately by root.
for e in entries:
 s=Path(e['source'])
 if not s.is_file():
  if s.name=='root-runtime-decision.json' or e['policy']=='EXTERNAL_ONLY':continue
  raise AssertionError('Missing required finalized source: '+str(s))
 if e['policy']!='EXTERNAL_ONLY':
  assert s.stat().st_size<=2_000_000,str(s)
  b=s.read_bytes();assert b.isascii(),str(s)
def pin_bytes(path,b):return {'path':str(path),'bytes':len(b),'sha256':sha(b)}
def write_new(target,b):
 target.parent.mkdir(parents=True,exist_ok=True)
 if target.exists():
  assert target.read_bytes()==b,('Existing target differs; do not overwrite',str(target))
  return
 with target.open('xb') as f:f.write(b)
 assert target.read_bytes()==b
for e in entries:
 s=Path(e['source']);row={k:v for k,v in e.items() if k not in ('exists_now','bytes_now','policy')};row['policy']=e['policy']
 if not s.is_file():
  row.update({'status':'OMITTED_ABSENT','reason':('Root will add final decision separately' if s.name=='root-runtime-decision.json' else e.get('reason','Absent external-only source'))});index.append(row);continue
 if s.stat().st_size>2_000_000 and s.name=='strata-paired-whole-graph-s106':
  audit=R/'paired-whole-graph-s106-prep/root-build-audit.json';d=json.loads(audit.read_text())
  row.update({'status':'EXTERNAL_ONLY','source_record':{'path':str(s),'bytes':s.stat().st_size,'sha256':d['build_sha256']['/out/strata-paired-whole-graph-s106'],'hash_provenance':str(audit),'new_binary_hash_performed':False}});index.append(row);continue
 assert s.stat().st_size<=2_000_000,('Unplanned large external file; no new hash',str(s))
 b=s.read_bytes();row['source_record']=pin_bytes(s,b)
 if e['policy']=='EXTERNAL_ONLY':
  row['source_is_ascii']=b.isascii();row['status']='EXTERNAL_ONLY';index.append(row);continue
 assert b.isascii(),str(s)
 t=REPO/e['target']
 if e['policy']=='EXACT_COPY':out=b;row['status']='EXACT_COPY_VERIFIED'
 else:
  data=json.loads(b)
  if s.name=='native-process-identity.json':
   assert isinstance(data,list)
   records=[]
   for record in data:
    assert 'proc_maps' in record and 'proc_maps_sha256' in record
    assert sha(record['proc_maps'].encode())==record['proc_maps_sha256']
    records.append({k:v for k,v in record.items() if k!='proc_maps'})
   derived={'record_type':'DERIVED_NATIVE_PROCESS_IDENTITY_SUMMARY','source_record':row['source_record'],'omitted_fields':['proc_maps'],'records':records}
  else:
   assert s.name in ('coding-0.json','coding-1.json')
   assert sha(data['text'].encode())==data['output_sha256']
   omitted=['chunks','text','content_events']
   retained={k:v for k,v in data.items() if k not in omitted}
   derived={'record_type':'DERIVED_REQUEST_SUMMARY','source_record':row['source_record'],'omitted_fields':omitted,'content_event_count':len(data.get('content_events',[])),'chunk_count':len(data.get('chunks',[])),'output_text_bytes':len(data['text'].encode()),'raw_source_on_disk':True,'summary':retained}
  out=(json.dumps(derived,indent=2,ensure_ascii=True)+'\n').encode();row['status']='DERIVED_VERIFIED_NOT_EXACT_COPY'
 assert out.isascii();write_new(t,out);row['target_record']=pin_bytes(e['target'],out)
 assert s.read_bytes()==b,'Source changed during copy: '+str(s)
 index.append(row)
record={'status':'CURATED_FINALIZED_RECORDS_WITH_ROOT_DECISION_ADDED_SEPARATELY','plan':{'path':str(PLAN),'sha256':EXPECTED},'scope':'Exact copies, explicitly derived compact records and external-only source index; no speed or closure inference by this copying script.','entries':index}
out=(json.dumps(record,indent=2)+'\n').encode();write_new(REPO/S106/'record-copies.json',out)
s104=[x for x in index if x.get('target','').startswith(S104+'/')]
write_new(REPO/S104/'record-copies.json',(json.dumps({'status':'EXACT_POST_REVIEW_COPIES','entries':s104},indent=2)+'\n').encode())
# Independent destination re-read against index after all writes.
for row in index:
 if 'target_record' not in row:continue
 tr=row['target_record'];b=(REPO/tr['path']).read_bytes();assert len(b)==tr['bytes'] and sha(b)==tr['sha256'] and b.isascii()
counts={k:sum(x['status']==k for x in index) for k in sorted({x['status'] for x in index})}
summary={'status':'PASS_FINAL_COPIES_AND_DERIVED_RECORDS_VERIFIED','counts':counts,'target_payload_bytes':sum(x.get('target_record',{}).get('bytes',0)for x in index),'s106_index_sha256':sha(out),'s106_index_bytes':len(out),'s104_index_entries':len(s104)}
(PREP/'curation-result.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
