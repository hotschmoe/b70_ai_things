"""Freeze recovered monitored-trial evidence; no deployment or GPU access."""
import json
from pathlib import Path
import serve


def prepare():
    base = serve.ROOT / 'results/bang_recurrence_testing_20260910'
    plan100 = base / 'native5802-backend100k-v2/plan.json'
    plan200 = base / 'native5802-backend200k/plan.json'
    receipt = plan100.parent / 'qualification-receipt.json'
    result = {'schema': 1, 'scope': serve.SCOPE, 'inputs_finalized': True,
        'user_authorized_monitored_trial': True, 'long200k_qualified': False,
        'unrestricted_model_quality_qualified': False,
        'plan100k': str(plan100), 'plan200k': str(plan200),
        'receipt100k': str(receipt),
        'receipt100k_sha256': '16a0b1f2ccd5f725e5a6802fb5603deeeb3a2f07de959f1f551cc9ed603b9c2d',
        'stopped200k_receipt': str(plan200.parent / 'parent-pause-outcome.json'),
        'recovery_receipt': str(base / 'native5802-userpause-recovery/parent-recovery-receipt.json'),
        'native_receipt': str(serve.REPO / 'vllm/int4/native_gdn_oracle/native5802_build/image-receipt.json'),
        'model_manifest': str(serve.ROOT / 'results/bang_isolation_20260910/sglang-calibrated-kv/current-verified-model-files.json'),
        'model_root': str(serve.REPO / 'models/files/qwen3.8-27b/int4-autoround-gptq-relabel-r212'),
        'known_quality_negative': str(plan100.parent / 'known-quality-negative.json')}
    files = serve.required_dependencies(result)
    files.update(serve.read(receipt)['evidence_hashes'])
    result['sha256'] = {p: serve.sha(p) for p in sorted(files)}
    serve.verify_inputs(result)
    if serve.INPUTS.exists():
        serve.require(serve.read(serve.INPUTS).get('inputs_finalized') is not True, 'already frozen; create separate revision')
    serve.INPUTS.write_text(json.dumps(result, indent=2, ensure_ascii=True) + '\n')
    print(json.dumps({'inputs_frozen': True, 'long200k_qualified': False,
                      'sha256': serve.sha(serve.INPUTS), 'files': len(files)}))

if __name__ == '__main__':
    prepare()
