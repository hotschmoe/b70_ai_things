import json,sys,pathlib
for n in sys.argv[1:]:
    p=pathlib.Path(n)/'bench.json'
    if not p.exists(): print(n,'missing'); continue
    d=json.loads(p.read_text()); out=[]
    for r in d['decode']:
        for i in range(len(r['engine_tps'])):
            t=r['tokens'][i]; pr,acc=r['drafts'][i]; w=t-acc
            out.append(f"c{r['c']} {r['engine_tps'][i]:.1f}t/s {t/w:.2f}tok/win {t/r['engine_tps'][i]/w*1000:.1f}ms/win {r['sha'][i][:8]}")
    pf=[f"{x['target']}:{x['prompt_tps']:.0f}" for x in d['prefill']]
    print(f"{n:14s} coh={d['coherence_ok']} | "+' | '.join(out)+' | pf '+' '.join(pf))
