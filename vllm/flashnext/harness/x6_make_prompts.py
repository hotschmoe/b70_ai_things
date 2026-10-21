"""Stable, fully tokenized raw prompts shared by Strata and vLLM gates."""
import hashlib
import json
import sys
from pathlib import Path
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained('/mnt/vm_8tb/b70/build/backend-decision/hf-flashnext', local_files_only=True, trust_remote_code=True)
unit = 'The history of computing is long and full of surprising turns. '
rows = []
for n in (4096, 16384, 32768):
    intro = 'Independent QSA reference X6. Deployment code is COBALT-7291.\n'
    suffix = '\nRepeat the deployment code, then give three short facts about Paris.\nAnswer:'
    ids = tok.encode(intro + unit * (n // 8), add_special_tokens=False)
    end = tok.encode(suffix, add_special_tokens=False)
    ids = ids[:n - len(end)] + end
    text = tok.decode(ids, skip_special_tokens=False)
    retok = tok.encode(text, add_special_tokens=False)
    assert ids == retok and len(ids) == n
    rows.append(dict(target_tokens=n, prompt=text, token_ids=ids,
                     prompt_sha256=hashlib.sha256(text.encode()).hexdigest(),
                     max_tokens=64, temperature=0))
Path(sys.argv[1]).write_text(json.dumps(rows, ensure_ascii=True, indent=2) + '\n')
print('X6_REFERENCE_PROMPTS_READY', [(r['target_tokens'], r['prompt_sha256']) for r in rows])
