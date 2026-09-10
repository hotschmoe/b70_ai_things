"""Freeze new plan/source identities; does not qualify, deploy or touch GPUs."""
import json
from pathlib import Path
import serve


def prepare():
    base=serve.ROOT/'results/bang_recurrence_testing_20260910'
    plans={k:base/('native5802-backend'+size)/'plan.json' for k,size in [('plan100k','100k-v2'),('plan200k','200k')]}
    for p in plans.values():
        serve.require(p.is_file() and (p.parent/'frozen.json').is_file(), 'new backend plan not yet frozen: '+str(p))
    native=serve.REPO/'vllm/int4/native_gdn_oracle/native5802_build/image-receipt.json'
    model_manifest=serve.ROOT/'results/bang_isolation_20260910/sglang-calibrated-kv/current-verified-model-files.json'
    negative=plans['plan100k'].parent/'known-quality-negative.json'
    tokenizer=base/'native5802-backend200k-tokenizer-v1/result.json'
    files=[Path(serve.__file__),Path(__file__),native,model_manifest,negative,tokenizer,
           serve.REPO/'vllm/fp8/openai_key_frontdoor.py']
    for p in plans.values():files += [p,p.parent/'frozen.json']
    result={'schema':1,'inputs_finalized':True,'scope':serve.SCOPE,'image':serve.IMAGE,'native_sha256':serve.NATIVE,
      **{k:str(v) for k,v in plans.items()},'native_receipt':str(native),'model_manifest':str(model_manifest),
      'model_root':str(serve.REPO/'models/files/qwen3.8-27b/int4-autoround-gptq-relabel-r212'),
      'known_quality_negative':str(negative),'tokenizer200k_receipt':str(tokenizer),'mtp_scale_layer':'mtp.layers.0.self_attn.attn',
      'required_jobs':{k:[j['name'] for j in serve.read(p)['jobs']] for k,p in plans.items()},
      'sha256':{str(p):serve.sha(p) for p in files}}
    serve.verify_inputs(result)
    if serve.INPUTS.exists():
        current=serve.read(serve.INPUTS)
        serve.require(current.get('inputs_finalized') is not True, 'inputs already frozen; prepare separate revision')
    serve.INPUTS.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'inputs_frozen':True,'qualified':False,'inputs_sha256':serve.sha(serve.INPUTS)}))

if __name__=='__main__':prepare()
