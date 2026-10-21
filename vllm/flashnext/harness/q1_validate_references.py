"""Host-side output gate. No model imports, weight loads or GPU access."""
import argparse
import hashlib
import json
from pathlib import Path

base = Path('/mnt/vm_8tb/b70/build/vllm-flashnext')
repo = Path('/mnt/vm_8tb/github/b70_ai_things')
p = argparse.ArgumentParser()
p.add_argument('--reference-dir', type=Path, default=base / 'q1')
a = p.parse_args()
cfg_path = a.reference_dir / 'strata-reference-config.json'
cfg = json.loads(cfg_path.read_text())
canonical_path = repo / 'strata/flash-next/night/configs/dd-nomtp-c1.json'
canonical = json.loads(canonical_path.read_text())
assert {k for k in set(cfg) | set(canonical) if cfg.get(k) != canonical.get(k)} == {'aliases', 'log'}
fixtures = json.loads((base / 'x6/reference-prompts.json').read_text())
rows = json.loads((a.reference_dir / 'strata-references.json').read_text())
assert len(rows) == len(fixtures) == 3
summaries = []
for f, row in zip(fixtures, rows, strict=True):
    assert row['prompt_sha256'] == f['prompt_sha256'] == hashlib.sha256(f['prompt'].encode()).hexdigest()
    assert row['prompt_tokens'] == len(f['token_ids']) == f['target_tokens']
    assert row['text_sha256'] == hashlib.sha256(row['text'].encode()).hexdigest()
    assert row['text'] and row['output_ids']
    assert len(row['raw_output_ids']) <= f['max_tokens']
    assert row['config'] == str(cfg_path)
    assert len(row['exe_sha256']) == 64
    info, last = row['engine_info'], row['engine_last']
    assert isinstance(info, dict) and info and isinstance(last, dict)
    assert last.get('drafts_offered') == 0, last
    assert last.get('drafts_accepted') == 0, last
    summaries.append(dict(prompt_tokens=row['prompt_tokens'], output_tokens=len(row['output_ids']),
                          drafts_offered=last['drafts_offered'], drafts_accepted=last['drafts_accepted']))
assert len({r['exe_sha256'] for r in rows}) == 1
summary = dict(canonical_config_sha256=hashlib.sha256(canonical_path.read_bytes()).hexdigest(),
               research_config_sha256=hashlib.sha256(cfg_path.read_bytes()).hexdigest(),
               exe_sha256=rows[0]['exe_sha256'], rows=summaries, no_draft_gate=True)
(a.reference_dir / 'strata-reference-summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=True) + '\n')
print('Q1_STRATA_REFERENCE_GATE_OK', json.dumps(summary, ensure_ascii=True))
