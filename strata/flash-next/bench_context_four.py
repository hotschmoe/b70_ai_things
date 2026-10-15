#!/usr/bin/env python3
"""Four distinct near-200K chats, two GPU slots, measured 32 GiB RAM parking.

Run through an already pair-leased server. Sample host RSS and container cgroup
RAM during this bounded test; do not start the persistent soak observer.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import subprocess
import threading
import time
import urllib.request

from bench_serving import get, send


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    x = (len(values)-1)*fraction
    i = int(x)
    return values[i]+(values[min(i+1, len(values)-1)]-values[i])*(x-i)


def classify_restores(rows, records, count):
    """Attribute only what native evidence proves; preserve unknown mixed cases."""
    if count!=len(records):
        return [None]*len(rows)
    if count==0:
        return [False]*len(rows)
    if count==len(rows):
        return [True]*len(rows)
    if count==1:
        matches=[i for i,r in enumerate(rows) if r['timings']['cache_n']==records[0]['tokens']]
        if len(matches)==1:
            return [i==matches[0] for i in range(len(rows))]
    return [None]*len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--base-url', default='http://127.0.0.1:18085')
    parser.add_argument('--target', type=int, default=199600)
    parser.add_argument('--cached-ttft-target-s', type=float, default=2.0)
    parser.add_argument('--resume', action='store_true', help='Continue saved phases on the same live engine PID/container')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=args.resume)
    base = args.base_url.rstrip('/')
    config = json.loads(args.config.read_text())
    argv = config['args']
    if int(argv[argv.index('--conversation-cache-mib')+1]) != 32768:
        raise RuntimeError('This qualification requires a configured 32768 MiB parking budget')
    container = json.loads(subprocess.check_output(['docker','inspect','strata-ab-serve']))[0]
    if str(args.config.resolve()) not in container['Config']['Cmd']:
        raise RuntimeError('Container is not using the supplied config')
    identity = get(base, '/v1/models')
    alias = 'qwen3.8-flash-next-ISTA-GSQ-RCO-IQ3_XXS-strata'
    ids = [r['id'] for r in identity['data']]
    registry = (Path(__file__).resolve().parents[2]/'evals/configs/models.yaml').read_text()
    if ids[0] != 'hotschmoe-dd' or alias not in ids or alias not in registry:
        raise RuntimeError('Model identity mismatch')
    if identity['data'][0]['meta']['n_ctx'] < 200000:
        raise RuntimeError('Two 200K slots required')
    (args.output/'models.json').write_text(json.dumps(identity,indent=2)+'\n')
    (args.output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    processes = subprocess.check_output(['docker','top','strata-ab-serve','-eo','pid,args'],text=True)
    native = [int(line.split(None,1)[0]) for line in processes.splitlines()[1:]
              if line.split(None,1)[1].split()[0] == config['exe']]
    if len(native) != 1:
        raise RuntimeError('Cannot identify the native engine PID')
    pid = native[0]
    meta_path=args.output/'run-meta.json'
    run_meta=dict(native_pid=pid,container_id=container['Id'],
                  config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
                  log_start=Path(config['log']).stat().st_size)
    if args.resume:
        prior=json.loads(meta_path.read_text())
        if any(prior[k]!=run_meta[k] for k in ['native_pid','container_id','config_sha256']):
            raise RuntimeError('Resume requires the original live engine and config')
        run_meta=prior
    else:
        meta_path.write_text(json.dumps(run_meta,indent=2)+'\n')
    progress_path=args.output/'progress.json'
    completed=json.loads(progress_path.read_text()) if args.resume else []
    cg_line = next(x for x in Path(f'/proc/{pid}/cgroup').read_text().splitlines() if x.startswith('0::'))
    cg = Path('/sys/fs/cgroup')/cg_line.split('::',1)[1].lstrip('/')
    if not (cg/'memory.current').exists():
        raise RuntimeError('Cannot sample the engine container memory cgroup')
    done = threading.Event()
    phase = 'baseline'
    memory = [json.loads(x) for x in (args.output/'memory.jsonl').read_text().splitlines() if x.strip()] if args.resume else []
    memory_errors = []
    def mem_fields():
        fields = dict(line.split() for line in (cg/'memory.stat').read_text().splitlines())
        events = dict(line.split() for line in (cg/'memory.events').read_text().splitlines())
        return dict(cgroup_anon_bytes=int(fields['anon']),cgroup_file_bytes=int(fields['file']),
                    cgroup_swap_bytes=int((cg/'memory.swap.current').read_text()),
                    cgroup_oom_kill=int(events.get('oom_kill',0)))
    def sample():
        with (args.output/'memory.jsonl').open('a' if args.resume else 'w',buffering=1) as output:
            while not done.is_set():
                try:
                    status = Path(f'/proc/{pid}/status').read_text()
                    rss = {k:int(v)*1024 for k,v in re.findall(r'^(VmRSS|VmHWM):\s+(\d+) kB',status,re.M)}
                    available = re.search(r'^MemAvailable:\s+(\d+) kB',Path('/proc/meminfo').read_text(),re.M)
                    row = dict(ts=time.time(),phase=phase,**rss,**mem_fields(),
                               cgroup_current_bytes=int((cg/'memory.current').read_text()),
                               cgroup_lifetime_peak_bytes=int((cg/'memory.peak').read_text()),
                               host_available_bytes=int(available[1])*1024)
                    memory.append(row);output.write(json.dumps(row)+'\n')
                except (OSError,ValueError) as exc:
                    memory_errors.append(str(exc));output.write(json.dumps(dict(error=str(exc)))+'\n')
                    return
                done.wait(.05)
    monitor = threading.Thread(target=sample,daemon=True)
    monitor.start()
    unit = 'The history of computing is long and full of surprising turns. '
    count = (args.target-180)//12
    chats = {}
    codes = {}
    all_rows = []
    phase_reports = []
    for label in 'ABCD':
        codes[label] = [f'{label}-COBALT-7291',f'{label}-RIVER-5183',f'{label}-ORBIT-9462']
        prompt = f'[Independent full conversation {label}] '
        previous = 0
        for i, fraction in enumerate([.1,.5,.95]):
            position = int(count*fraction)
            prompt += unit*(position-previous)+f'\nDeployment code {i+1} is {codes[label][i]}.\n'
            previous = position
        prompt += unit*(count-previous)+'\nReturn only deployment codes 1, 2 and 3, in order.'
        chats[label] = [dict(role='user',content=prompt)]
    for saved in completed:
        phase_reports.append(json.loads((args.output/(saved['phase']+'-overlap.json')).read_text()))
        for label in saved['labels']:
            row=json.loads((args.output/(saved['name']+'-'+label+'.json')).read_text())
            row.setdefault('retained_history_reused',row['retained_prompt_reused'] and row['prior_tail_replayed_tokens']==0)
            all_rows.append(row)
            chats[label]=row['request']['messages']+[dict(role='assistant',content=row['text']),
                dict(role='user',content='Return those same three deployment codes again, in order. Output only the codes.')]
    def prepare(label, name):
        messages = chats[label]
        resident = name.startswith('resident')
        if resident:
            # The final user message is new, unsent input. Prior history stays intact.
            messages = messages[:-1]+[dict(role='user',content=
                'Return those same three deployment codes in order on the first line. '
                'Then write at least 300 words explaining paged KV caches and RAM eviction. '
                'Keep the codes accurate; give only this answer.')]
        cap = 256 if resident else 64
        body = dict(model='hotschmoe-dd',messages=messages,thinking=dict(type='disabled'),max_tokens=cap)
        req = urllib.request.Request(base+'/v1/messages/count_tokens',json.dumps(body).encode(),
                                     {'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=30) as response:
            prompt_tokens = json.load(response)['input_tokens']
        if prompt_tokens+cap>200000-8:
            raise RuntimeError('Rendered prompt plus requested output exceeds the 200K budget')
        return messages,cap,prompt_tokens

    def record(label, name, restore, prepared):
        messages,cap,prompt_tokens = prepared
        resident = name.startswith('resident')
        marks = {}
        row = send(base,'hotschmoe-dd',messages,cap,
                   on_first_content=lambda:marks.update(first=time.perf_counter()))
        if row['usage']['prompt_tokens']!=prompt_tokens:
            raise RuntimeError('Preflight and generation templates tokenized differently')
        # Align the retained per-event monotonic offsets to this process's common clock.
        origin = marks['first']-row['content_events'][0]['elapsed_s']
        row['first_content_mono_s'] = origin+row['content_events'][0]['elapsed_s']
        row['last_content_mono_s'] = origin+row['content_events'][-1]['elapsed_s']
        if row['usage']['prompt_tokens']+row['usage']['completion_tokens']>200000:
            raise RuntimeError('The real rendered conversation exceeded its 200K slot')
        if resident and row['usage']['completion_tokens']<128:
            raise RuntimeError('Resident reply too short for a meaningful overlap test')
        previous=next((r for r in reversed(all_rows) if r['conversation']==label),None)
        previous_prompt=previous['usage']['prompt_tokens'] if previous else 0
        # The engine normally holds all but its last emitted token. Keep any
        # additional replay visible, including reply-token/template seams.
        previous_held=max(0,previous['usage']['total_tokens']-1) if previous else 0
        row.update(conversation=label,phase=name,expected_restore=restore,
                   previous_prompt_tokens=previous_prompt,previous_held_tokens=previous_held,
                   retained_prompt_reused=not previous or row['timings']['cache_n']>=previous_prompt-1,
                   prior_tail_replayed_tokens=max(0,previous_held-row['timings']['cache_n']))
        row['retained_history_reused']=row['retained_prompt_reused'] and row['prior_tail_replayed_tokens']==0
        (args.output/(name+'-'+label+'.json')).write_text(json.dumps(row,indent=2)+'\n')
        text = row['text']
        if not all(code in text for code in codes[label]):
            raise RuntimeError(name+' '+label+' lost its needles: '+text)
        if any(code in text for other in 'ABCD' if other!=label for code in codes[other]):
            raise RuntimeError('Cross-conversation state leak')
        if row['usage']['prompt_tokens'] < args.target-1000:
            raise RuntimeError('Context too short')
        if name!='initial' and (not row['retained_history_reused'] or row['timings']['prompt_n']>128):
            raise RuntimeError('Returning full conversation recomputed retained history')
        chats[label] = messages+[dict(role='assistant',content=text),dict(role='user',content=
            'Return those same three deployment codes again, in order. Output only the codes.')]
        print(json.dumps(dict(phase=name,conversation=label,ttft_s=row['ttft_s'],wall_s=row['wall_s'],
                              timings=row['timings'],text=text)),flush=True)
        return row
    def pair(labels,name,restore):
        nonlocal phase
        phase=name+'-'+labels
        if any(x['phase']==phase for x in completed):
            return
        prepared={label:prepare(label,name) for label in labels}
        before=get(base,'/metrics')['conversation_cache']
        phase_log_start=log.stat().st_size
        start=time.perf_counter()
        with ThreadPoolExecutor(max_workers=2) as pool:
            rows=list(pool.map(lambda label:record(label,name,restore,prepared[label]),labels))
        all_rows.extend(rows)
        slots=get(base,'/slots');metrics=get(base,'/metrics')
        (args.output/(phase+'-state.json')).write_text(json.dumps(dict(slots=slots,metrics=metrics,
            wall_s=time.perf_counter()-start),indent=2)+'\n')
        if len(slots)!=2 or any(slot['n_prompt_tokens']<args.target-1000 for slot in slots):
            raise RuntimeError('Both GPU slots must retain near-200K conversations')
        cache=metrics['conversation_cache']
        restored=cache['restores']-before['restores']
        # stderr cache/batch summaries can arrive just after HTTP completion.
        deadline=time.monotonic()+2
        while True:
            phase_log=log.read_text()[phase_log_start:]
            batch_stats=re.findall(r'strata batch: (\d+) windows, avg ([\d.]+) rows',phase_log)
            if (not restore or restored>=2) and (not name.startswith('resident') or batch_stats):
                break
            if time.monotonic()>=deadline:
                break
            time.sleep(.05)
            cache=get(base,'/metrics')['conversation_cache']
            restored=cache['restores']-before['restores']
        restore_records=[dict(tokens=int(n),elapsed_ms=float(ms),payload_bytes=int(nb))
                         for n,ms,nb in re.findall(
                             r'conversation cache: restored (\d+) tokens.*? in ([\d.]+) ms;[^\n]* restored_payload_bytes=(\d+)',
                             phase_log)]
        overlap=min(row['last_content_mono_s'] for row in rows)-max(row['first_content_mono_s'] for row in rows)
        report=dict(phase=phase,content_interval_overlap_s=overlap,
                    native_batch_windows=[dict(windows=int(n),emitted_rows_per_window=float(nr))
                                          for n,nr in batch_stats],actual_ram_restores=restored,
                    snapshot_restore_records=restore_records,
                    snapshot_restore_payload_bytes=sum(r["payload_bytes"] for r in restore_records))
        phase_reports.append(report)
        (args.output/(phase+'-overlap.json')).write_text(json.dumps(report,indent=2)+'\n')
        (args.output/(phase+'-engine.log')).write_text(phase_log)
        # Each MTP slot emits <=2 tokens/window. An average above2 requires
        # actual windows with two live decode slots; HTTP overlap alone is weaker.
        if name.startswith('resident') and (overlap<.1 or not any(float(nr)>2 for _,nr in batch_stats)):
            raise RuntimeError('Two near-full active decodes were not proven: '+json.dumps(report))
        report['intended_ram_exchange']=restore
        report['forced_exchange_passed']=not restore or (restored==2 and len(restore_records)==2
            and all(r['tokens']>=args.target-1000 and r['payload_bytes']>0 for r in restore_records))
        report['restore_classification']='resident_only' if restored==0 else 'ram_only' if restored==2 else 'mixed'
        # Idle-slot handoff can park one return before its partner is admitted.
        # Classify observed restores; an intended resident pair need not stay resident.
        actual_restores=classify_restores(rows,restore_records,restored)
        for row,actual in zip(rows,actual_restores):
            row['actual_ram_restore']=actual
            (args.output/(name+'-'+row['conversation']+'.json')).write_text(json.dumps(row,indent=2)+'\n')
        if cache['budget_mib']!=32768:
            raise RuntimeError('Live parking budget differs from requested 32 GiB')
        (args.output/(phase+'-overlap.json')).write_text(json.dumps(report,indent=2)+'\n')
        completed.append(dict(phase=phase,labels=labels,name=name))
        temporary=progress_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(completed,indent=2)+'\n');temporary.replace(progress_path)
    log=Path(config['log']);log_start=run_meta['log_start']
    try:
        pair('AB','initial',False)
        pair('AB','resident1',False)
        (args.output/'two-active-summary.json').write_text(json.dumps(dict(
            correctness_passed=True,overlap=next(p for p in phase_reports if p['phase']=='resident1-AB'),
            actual_prompt_tokens=[r['usage']['prompt_tokens'] for r in all_rows
                                  if r['conversation'] in 'AB' and r['phase'] in ('initial','resident1')],
            note='Two near-full active decodes proven before introducing C/D.'),indent=2)+'\n')
        # Add C/D without repeating A/B's huge prefills. Both pairs retain all turns.
        pair('CD','initial',False)
        pair('CD','resident1',False)
        pair('AB','exchange1',True)
        pair('CD','exchange1',True)
        phase='complete'
    finally:
        done.set();monitor.join(timeout=5)
        (args.output/'engine-segment.log').write_text(log.read_text()[log_start:])
    if memory_errors or not memory:
        raise RuntimeError('RAM telemetry unavailable: '+repr(memory_errors))
    segment=(args.output/'engine-segment.log').read_text()
    restores=re.findall(r'conversation cache: restored (\d+) tokens.*? in ([\d.]+) ms;[^\n]* restored_payload_bytes=(\d+)',segment)
    full_restores=[dict(tokens=int(n),elapsed_ms=float(ms),payload_bytes=int(nb))
                   for n,ms,nb in restores if int(n)>=args.target-1000]
    forced_exchange_passed=len(full_restores)>=4 and all(p['forced_exchange_passed'] for p in phase_reports if p['intended_ram_exchange'])
    cached=[r for r in all_rows if r['phase']!='initial']
    categories={}
    for kind,rows in [('resident',[r for r in cached if r['actual_ram_restore'] is False]),
                      ('ram_restore',[r for r in cached if r['actual_ram_restore'] is True]),
                      ('unattributed_mixed_pair',[r for r in cached if r['actual_ram_restore'] is None])]:
        values=[r['ttft_s'] for r in rows]
        categories[kind]=dict(samples=len(rows),ttft_p50_s=percentile(values,.5),ttft_p95_s=percentile(values,.95),
                              p95_target_s=args.cached_ttft_target_s,
                              latency_target_met=percentile(values,.95)<=args.cached_ttft_target_s if values else None)
    summary=dict(correctness_passed=True,four_distinct=True,forced_ram_exchange_passed=forced_exchange_passed,configured_parking_gib=32,
                 actual_prompt_tokens=[r['usage']['prompt_tokens'] for r in all_rows if r['phase']=='initial'],
                 cached_return_latency=categories,native_full_restores=full_restores,
                 retained_prompt_history_reused=all(r['retained_prompt_reused'] for r in cached),
                 retained_history_reused=all(r['retained_history_reused'] for r in cached),
                 maximum_prior_tail_replayed_tokens=max(r['prior_tail_replayed_tokens'] for r in cached),
                 cached_per_conversation={label:dict(
                     samples=len([r for r in cached if r['conversation']==label]),
                     ttft_p50_s=percentile([r['ttft_s'] for r in cached if r['conversation']==label],.5),
                     ttft_p95_s=percentile([r['ttft_s'] for r in cached if r['conversation']==label],.95),
                     requests=[dict(phase=r['phase'],ttft_s=r['ttft_s'],wall_s=r['wall_s'],
                                    client_decode_tps=r['client_decode_tps'],
                                    gaps_ms=r['content_event_gap_ms'],timings=r['timings'])
                               for r in cached if r['conversation']==label]) for label in 'ABCD'},
                 phase_reports=phase_reports,
                 total_context_tokens_at_end={label:next(r['usage']['total_tokens'] for r in reversed(all_rows)
                                                        if r['conversation']==label) for label in 'ABCD'},
                 sampled_peak_native_rss_bytes=max(r['VmRSS'] for r in memory),
                 native_hwm_before_bytes=memory[0]['VmHWM'],
                 native_hwm_after_bytes=memory[-1]['VmHWM'],
                 sampled_peak_exchange_native_rss_bytes=max(r['VmRSS'] for r in memory if r['phase'].startswith('exchange')),
                 sampled_peak_exchange_container_bytes=max(r['cgroup_current_bytes'] for r in memory if r['phase'].startswith('exchange')),
                 container_memory_at_sampled_peak=max(memory,key=lambda r:r['cgroup_current_bytes']),
                 cgroup_oom_kill_delta=memory[-1]['cgroup_oom_kill']-memory[0]['cgroup_oom_kill'],
                 cgroup_swap_before_bytes=memory[0]['cgroup_swap_bytes'],
                 cgroup_swap_max_bytes=max(r['cgroup_swap_bytes'] for r in memory),
                 sampled_peak_container_ram_bytes=max(r['cgroup_current_bytes'] for r in memory),
                 container_lifetime_peak_bytes=max(r['cgroup_lifetime_peak_bytes'] for r in memory),
                 minimum_host_available_bytes=min(r['host_available_bytes'] for r in memory),
                 ram_sample_period_s=.05,ram_samples=len(memory),native_pid=pid,
                 config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
                 note='Empirical p95 on bounded samples; sampled exchange peaks and lifetime cgroup peak are separate. '
                      'Restore bytes are snapshot restore API payload, not physical PCIe traffic; '
                      'native records belong to each request pair and are not assigned to individual streams.')
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)

if __name__=='__main__':
    main()
