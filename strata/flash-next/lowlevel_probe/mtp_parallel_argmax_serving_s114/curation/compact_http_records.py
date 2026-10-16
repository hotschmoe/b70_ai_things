from pathlib import Path
import hashlib
import json

R = Path('/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014')
REPO = Path('/mnt/vm_8tb/github/b70_ai_things')
S = R/'mtp-parallel-argmax-s114-serving-prep-v2'
D = REPO/'strata/flash-next/lowlevel_probe/mtp_parallel_argmax_serving_s114'
index = json.loads((D/'record-copies.json').read_text())
copies = index['copies']
derived = index['derived']

def sha(b):
    return hashlib.sha256(b).hexdigest()

def write_json(p,j):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(j,indent=2,ensure_ascii=True)+'\n',encoding='ascii')

comparison = json.loads((S/'comparison.json').read_text())
assert comparison['status']=='PASS_MATCHED_DISCRIMINATOR'
assert all(comparison['checks'].values())
raw_records=[]
for p in sorted(S.rglob('*')):
    if not p.is_file(): continue
    relative = p.relative_to(S)
    if relative.parts[0] not in ('mode0','mode1') and str(relative) not in ('comparison.json','serving-inputs.json'):
        continue
    if p.name=='engine.log': continue  # final engine log is already copied
    b=p.read_bytes()
    assert b.isascii() and b'\0' not in b,str(p)
    t=D/'serving-actual'/relative
    if p.name in ('coding-0.json','coding-1.json'):
        j=json.loads(b)
        contents=''.join(c.get('delta',{}).get('content','') or '' for chunk in j['chunks'] for c in chunk.get('choices',[]))
        assert contents==j['text']
        assert sha(contents.encode('utf-8'))==j['output_sha256']
        view={k:v for k,v in j.items() if k not in ('chunks','content_events')}
        view.update({'SSE_concatenated_content':contents,
            'SSE_content_exact_text':True,'original_chunk_count':len(j['chunks']),
            'original_content_event_count':len(j['content_events']),
            'curation_scope':'Full request, text, concatenated SSE content, usage, timing/work, lifetime and event-gap metrics retained. Raw chunk objects, response IDs and event timestamps omitted; no chunk segmentation or raw ID equality claim.',
            'external_original':{'path':str(p),'bytes':len(b),'sha256':sha(b)}})
        t=t.with_name(t.stem+'-compact.json')
        write_json(t,view)
        tb=t.read_bytes()
        entry={'target':str(t.relative_to(REPO)),'bytes':len(tb),'sha256':sha(tb),
            'source':str(p),'source_bytes':len(b),'source_sha256':sha(b),
            'derivation':'add_runtime.py: remove chunks/content_events, retain all other fields, append literal ordered choices[].delta.content concatenation and event counts; assert concatenation equals complete text/output SHA.'}
        derived.append(entry)
        raw_records.append(view['external_original'])
    else:
        assert len(b)<100000,str(p)
        t.parent.mkdir(parents=True,exist_ok=True);t.write_bytes(b)
        copies.append({'source':str(p),'target':str(t.relative_to(REPO)),'bytes':len(b),'sha256':sha(b)})

script=Path(__file__)
b=script.read_bytes()
t=D/'curation/compact_http_records.py'
t.parent.mkdir(parents=True,exist_ok=True);t.write_bytes(b)
derived.append({'target':str(t.relative_to(REPO)),'bytes':len(b),'sha256':sha(b),
    'source':str(script),'derivation':'Exact curation script for deterministic bounded HTTP view; repository copy is provenance, not a serving controller.'})
index.update({'status':'SOURCE_BUILD_AUDIT_RUNTIME_CLOSED_POSTREVIEWS_PENDING',
    'pending_runtime':[], 'pending_postreviews':'Root actual post-model calls and decision not yet complete; no inferred final default/promotion decision.',
    'external_raw_HTTP_records':raw_records})
write_json(D/'record-copies.json',index)
write_json(R/'mtp-parallel-argmax-s114-curation/runtime-copy-status.json',{
    'status':index['status'],'exact_copies':len(copies),'exact_copy_bytes':sum(e['bytes'] for e in copies),
    'derived_records':len(derived),'external_raw_HTTP_records':len(raw_records),
    'external_raw_HTTP_bytes':sum(e['bytes'] for e in raw_records),
    'no_binary_archive_object_model_trace_or_GPU_read':True})
print((R/'mtp-parallel-argmax-s114-curation/runtime-copy-status.json').read_text())
