#!/usr/bin/env python3
"""GPU oracles for the two 0.2.0 re-ports (#2526 slot parking, #2511 batch draft counts).

Run against a server started with strata/flash-next/night/nl.py and a "parallel": 2 config with batch MTP
(pr020-check/configs/*.json).  Talks to 127.0.0.1:$NL_PORT (default 18085), never touches the GPU itself.

  slot_oracle.py park   --log ENGINE_LOG --out DIR [--tokens 12000]
  slot_oracle.py drafts --log ENGINE_LOG --out DIR [--delay 4]

park:   A and B decode 256-token replies in the two slots; C and D are then admitted into those slots (the
        overwrite); A and B return one at a time with their replies kept verbatim.  The returning turn must
        reuse everything the engine held after the reply (prompt + reply - 1 tokens): replay = held - cache_n.
        Fixed build: replay 0 for A and B, a "parked N tokens from slot S" line per overwritten slot.
        Unfixed main: replay ~255 (the reply re-read) or the whole prefix (the copies evicted).
drafts: (1) a 512-token request decodes alone, a 64-token one arrives --delay s later: the first is promoted into
        a slot, then goes back to the solo path when the short one ends (#1869); (2) a 256/256 pair.  The
        requests' timings.draft_n / draft_n_accepted are compared with the sum of the engine's per-DONE
        "drafts accepted X of Y" lines in the same log window and with the /metrics totals delta.
        Without batch MTP (SYCL with a layer split turns it off) every BDONE says 0 0, so the fixed build's
        API sums EQUAL the engine sums; unfixed main drops the legs before the final DONE (API < engine).
"""
import argparse, json, re, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, '/mnt/vm_8tb/github/b70_ai_things/strata/flash-next/night')
import nl  # noqa: E402  (send/get against NL_PORT)

UNIT = 'The history of computing is long and full of surprising turns. '


def chat(label, tokens):
    codes = [f'{label}-COBALT-{7000 + i}' for i in range(3)]
    n = max(3, tokens // 12)
    body = f'[Independent conversation {label}] '
    for i, frac in enumerate((.1, .5, .9)):
        body += UNIT * int(n * frac / 3) + f'\nDeployment code {i + 1} is {codes[i]}.\n'
    body += UNIT * (n // 3) + '\nRepeat the three deployment codes in order, then explain paged KV caches in 300 words.'
    return [dict(role='user', content=body)], codes


def pair(msgs, caps):
    with ThreadPoolExecutor(len(msgs)) as ex:
        return list(ex.map(lambda mc: nl.send(mc[0], mc[1]), zip(msgs, caps)))


def held(row):
    u = row['usage']
    return u['prompt_tokens'] + u['completion_tokens'] - 1


def cmd_park(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    log = Path(a.log); log0 = log.stat().st_size if log.exists() else 0
    ids = [m['id'] for m in nl.get('/v1/models')['data']]
    convs = {k: chat(k, a.tokens) for k in 'ABCD'}
    hist, rows = {}, {}
    for labels in ('AB', 'CD'):   # AB decode in the slots, then CD overwrite those slots
        res = pair([convs[k][0] for k in labels], [256, 256])
        for k, r in zip(labels, res):
            rows[k] = r
            hist[k] = convs[k][0] + [dict(role='assistant', content=r['text'])]
            print(json.dumps(dict(phase='first', conv=k, usage=r['usage'], timings=r['timings'])), flush=True)
    time.sleep(1)
    verdict = dict(models=ids, returns={})
    for k in 'AB':                # one at a time: no admission races, a clean cache_n per return
        msgs = hist[k] + [dict(role='user', content='Return the three deployment codes again, in order. Codes only.')]
        r = nl.send(msgs, 32)
        t = r['timings'] or {}
        replay = max(0, held(rows[k]) - int(t.get('cache_n') or 0))
        ok = replay == 0 and all(c in r['text'] for c in convs[k][1])
        verdict['returns'][k] = dict(held_before=held(rows[k]), cache_n=t.get('cache_n'), prompt_n=t.get('prompt_n'),
                                     replayed=replay, needles=all(c in r['text'] for c in convs[k][1]), ok=ok,
                                     text=r['text'][:200])
        print(json.dumps(dict(phase='return', conv=k, **verdict['returns'][k])), flush=True)
    seg = log.read_text(errors='replace')[log0:] if log.exists() else ''
    verdict['slot_parks'] = re.findall(r'conversation cache: (parked|skipped) (\d+) tokens from slot (\d+)', seg)
    verdict['slot_park_skips'] = re.findall(r'slot \d+ not parked before its overwrite \([^)]*\)', seg)
    verdict['metrics_cache'] = nl.get('/metrics').get('conversation_cache')
    verdict['passed'] = all(v['ok'] for v in verdict['returns'].values())
    (out / 'park-verdict.json').write_text(json.dumps(verdict, indent=2) + '\n')
    (out / 'park-engine-segment.log').write_text(seg)
    print(json.dumps(dict(passed=verdict['passed'], slot_parks=verdict['slot_parks'])), flush=True)
    return 0 if verdict['passed'] else 1


def totals():
    t = nl.get('/metrics')['totals']
    return int(t['drafts_accepted']), int(t['drafts_offered'])


DONE_DRAFTS = re.compile(r'strata serve: prompt \d+ tokens = .*? drafts accepted (\d+) of (\d+)')


def cmd_drafts(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    log = Path(a.log)
    whole = log.read_text(errors='replace')
    batch_mtp_off = 'WARNING: --batch-mtp is off' in whole or 'batch-mtp' not in whole
    verdict = dict(models=[m['id'] for m in nl.get('/v1/models')['data']], batch_mtp_off=batch_mtp_off, cases={})
    msgs = [[dict(role='user', content=p)] for p in (nl.CODING, nl.PROSE)]
    for name, caps, delay in (('staggered_promote_handback', [512, 64], a.delay), ('both_in_slots', [256, 256], 0.0)):
        a0, o0 = totals()
        log0 = log.stat().st_size
        with ThreadPoolExecutor(2) as ex:
            f0 = ex.submit(nl.send, msgs[0], caps[0])
            time.sleep(delay)   # > 0: the first request decodes alone first, then is promoted to a slot (#1869)
            f1 = ex.submit(nl.send, msgs[1], caps[1])
            res = [f0.result(), f1.result()]
        time.sleep(1)
        a1, o1 = totals()
        seg = log.read_text(errors='replace')[log0:]
        eng = [(int(x), int(y)) for x, y in DONE_DRAFTS.findall(seg)]
        reqs = [dict(cap=c, completion=r['usage']['completion_tokens'],
                     draft_n=(r['timings'] or {}).get('draft_n'),
                     draft_n_accepted=(r['timings'] or {}).get('draft_n_accepted')) for c, r in zip(caps, res)]
        api = (sum(int(q['draft_n_accepted'] or 0) for q in reqs), sum(int(q['draft_n'] or 0) for q in reqs))
        eng_sum = (sum(x for x, _ in eng), sum(y for _, y in eng))
        case = dict(requests=reqs, engine_done_lines=eng, engine_done_sum=eng_sum, api_sum=api,
                    metrics_delta=(a1 - a0, o1 - o0), metrics_match_api=(a1 - a0, o1 - o0) == api,
                    # without batch MTP every slot leg reports 0 0, so the requests' totals are exactly the sum of the
                    # engine's DONE lines; with it the BDONE counts come on top (not in the log)
                    engine_match=api == eng_sum if batch_mtp_off else (api[0] >= eng_sum[0] and api[1] >= eng_sum[1]),
                    sane=all((q['draft_n_accepted'] or 0) <= (q['draft_n'] or 0) for q in reqs))
        verdict['cases'][name] = case
        (out / f'drafts-{name}-engine.log').write_text(seg)
        print(json.dumps(dict(case=name, **case)), flush=True)
    verdict['passed'] = all(c['metrics_match_api'] and c['engine_match'] and c['sane'] for c in verdict['cases'].values())
    (out / 'drafts-verdict.json').write_text(json.dumps(verdict, indent=2) + '\n')
    print(json.dumps(dict(passed=verdict['passed'])), flush=True)
    return 0 if verdict['passed'] else 1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = p.add_subparsers(dest='cmd', required=True)
    x = s.add_parser('park'); x.add_argument('--log', required=True); x.add_argument('--out', required=True)
    x.add_argument('--tokens', type=int, default=12000)
    x = s.add_parser('drafts'); x.add_argument('--log', required=True); x.add_argument('--out', required=True)
    x.add_argument('--delay', type=float, default=4.0)
    a = p.parse_args()
    sys.exit(cmd_park(a) if a.cmd == 'park' else cmd_drafts(a))


if __name__ == '__main__':
    main()
