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

from bench_serving import get, send


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    x = (len(values)-1)*fraction
    i = int(x)
    return values[i]+(values[min(i+1, len(values)-1)]-values[i])*(x-i)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--base-url', default='http://127.0.0.1:18085')
    parser.add_argument('--target', type=int, default=199600)
    parser.add_argument('--cached-ttft-target-s', type=float, default=2.0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
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
    cg_line = next(x for x in Path(f'/proc/{pid}/cgroup').read_text().splitlines() if x.startswith('0::'))
    cg = Path('/sys/fs/cgroup')/cg_line.split('::',1)[1].lstrip('/')
    if not (cg/'memory.current').exists():
        raise RuntimeError('Cannot sample the engine container memory cgroup')
    done = threading.Event()
    phase = 'baseline'
    memory = []
    memory_errors = []
    def sample():
        with (args.output/'memory.jsonl').open('w',buffering=1) as output:
            while not done.is_set():
                try:
                    status = Path(f'/proc/{pid}/status').read_text()
                    rss = {k:int(v)*1024 for k,v in re.findall(r'^(VmRSS|VmHWM):\s+(\d+) kB',status,re.M)}
                    available = re.search(r'^MemAvailable:\s+(\d+) kB',Path('/proc/meminfo').read_text(),re.M)
                    row = dict(ts=time.time(),phase=phase,**rss,
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
    def record(label, name, restore):
        row = send(base,'hotschmoe-dd',chats[label],96)
        row.update(conversation=label,phase=name,expected_restore=restore)
        (args.output/(name+'-'+label+'.json')).write_text(json.dumps(row,indent=2)+'\n')
        text = row['text']
        if not all(code in text for code in codes[label]):
            raise RuntimeError(name+' '+label+' lost its needles: '+text)
        if any(code in text for other in 'ABCD' if other!=label for code in codes[other]):
            raise RuntimeError('Cross-conversation state leak')
        if row['usage']['prompt_tokens'] < args.target-1000:
            raise RuntimeError('Context too short')
        if name!='initial' and (row['timings']['cache_n'] < args.target-1000 or row['timings']['prompt_n']>128):
            raise RuntimeError('Returning full conversation recomputed retained history')
        chats[label] = chats[label]+[dict(role='assistant',content=text),dict(role='user',content=
            'Return those same three deployment codes again, in order. Output only the codes.')]
        print(json.dumps(dict(phase=name,conversation=label,ttft_s=row['ttft_s'],wall_s=row['wall_s'],
                              timings=row['timings'],text=text)),flush=True)
        return row
    def pair(labels,name,restore):
        nonlocal phase
        phase=name+'-'+labels
        before=get(base,'/metrics')['conversation_cache']
        start=time.perf_counter()
        with ThreadPoolExecutor(max_workers=2) as pool:
            rows=list(pool.map(lambda label:record(label,name,restore),labels))
        all_rows.extend(rows)
        slots=get(base,'/slots');metrics=get(base,'/metrics')
        (args.output/(phase+'-state.json')).write_text(json.dumps(dict(slots=slots,metrics=metrics,
            wall_s=time.perf_counter()-start),indent=2)+'\n')
        if len(slots)!=2 or any(slot['n_prompt_tokens']<args.target-1000 for slot in slots):
            raise RuntimeError('Both GPU slots must retain near-200K conversations')
        cache=metrics['conversation_cache']
        restored=cache['restores']-before['restores']
        if name!='initial' and restored!=(2 if restore else 0):
            raise RuntimeError('Expected restore classification is not supported by native cache counters: '+str(restored))
        for row in rows:
            row['actual_ram_restore']=name!='initial' and restored==2
            (args.output/(name+'-'+row['conversation']+'.json')).write_text(json.dumps(row,indent=2)+'\n')
        if cache['budget_mib']!=32768:
            raise RuntimeError('Live parking budget differs from requested 32 GiB')
    log=Path(config['log']);log_start=log.stat().st_size
    try:
        pair('AB','initial',False)
        pair('CD','initial',False)
        # CD are resident; AB must return from RAM, then displace CD. Repeat
        # both directions to qualify all four histories rather than one reused pair.
        pair('CD','resident1',False)
        pair('AB','exchange1',True)
        pair('AB','resident1',False)
        pair('CD','exchange1',True)
        for turn in [2,3]:
            pair('AB','exchange'+str(turn),True)
            pair('CD','exchange'+str(turn),True)
        phase='complete'
    finally:
        done.set();monitor.join(timeout=5)
        (args.output/'engine-segment.log').write_text(log.read_text()[log_start:])
    if memory_errors or not memory:
        raise RuntimeError('RAM telemetry unavailable: '+repr(memory_errors))
    segment=(args.output/'engine-segment.log').read_text()
    restores=re.findall(r'conversation cache: restored (\d+) tokens.*? in ([\d.]+) ms;',segment)
    full_restores=[(int(n),float(ms)) for n,ms in restores if int(n)>=args.target-1000]
    if len(full_restores)<12:
        raise RuntimeError('Missing native RAM-restore evidence for the twelve exchanged returns')
    cached=[r for r in all_rows if r['phase']!='initial']
    categories={}
    for kind,rows in [('resident',[r for r in cached if not r['expected_restore']]),
                      ('ram_restore',[r for r in cached if r['expected_restore']])]:
        values=[r['ttft_s'] for r in rows]
        categories[kind]=dict(samples=len(rows),ttft_p50_s=percentile(values,.5),ttft_p95_s=percentile(values,.95),
                              p95_target_s=args.cached_ttft_target_s,
                              latency_target_met=percentile(values,.95)<=args.cached_ttft_target_s)
    summary=dict(correctness_passed=True,four_distinct=True,configured_parking_gib=32,
                 actual_prompt_tokens=[r['usage']['prompt_tokens'] for r in all_rows if r['phase']=='initial'],
                 cached_return_latency=categories,native_full_restores=full_restores,
                 sampled_peak_native_rss_bytes=max(r['VmRSS'] for r in memory),
                 sampled_peak_container_ram_bytes=max(r['cgroup_current_bytes'] for r in memory),
                 container_lifetime_peak_bytes=max(r['cgroup_lifetime_peak_bytes'] for r in memory),
                 minimum_host_available_bytes=min(r['host_available_bytes'] for r in memory),
                 ram_sample_period_s=.05,ram_samples=len(memory),native_pid=pid,
                 config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
                 note='Empirical p95 on bounded samples; sampled exchange peaks and lifetime cgroup peak are separate.')
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)

if __name__=='__main__':
    main()
