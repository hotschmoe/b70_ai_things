"""Cache a revision-pinned WikiText-2 test prefix, without any GPU import."""
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import pyarrow.parquet as pq
from tokenizers import Tokenizer

out = Path('/mnt/vm_8tb/b70/build/vllm-flashnext/x5')
with urlopen('https://huggingface.co/api/datasets/Salesforce/wikitext', timeout=30) as r:
    metadata = json.load(r)
revision = metadata['sha']
url = f'https://huggingface.co/datasets/Salesforce/wikitext/resolve/{revision}/wikitext-2-raw-v1/test-00000-of-00001.parquet'
path = out / 'wikitext-2-test.parquet'
with urlopen(url, timeout=60) as r:
    payload = r.read()
path.write_bytes(payload)
text = '\n'.join(pq.read_table(path, columns=['text']).column('text').to_pylist())
tokenizer_path = Path('/mnt/cache/b70/devmodel/Qwen/Qwen3.6-35B-A3B/tokenizer.json')
tok = Tokenizer.from_file(str(tokenizer_path))
ids = tok.encode(text, add_special_tokens=False).ids[:2048]
assert len(ids) == 2048
record = dict(
    label='WikiText-2 raw-v1 test prefix, four independent 512-token chunks; not full-test PPL',
    source='https://huggingface.co/datasets/Salesforce/wikitext',
    revision=revision, source_file=url, license=metadata.get('cardData', {}).get('license'),
    parquet_sha256=hashlib.sha256(payload).hexdigest(),
    tokenizer_sha256=hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
    token_ids_sha256=hashlib.sha256(json.dumps(ids).encode('ascii')).hexdigest(),
    passages=[{'token_ids': ids[i:i+512]} for i in range(0, 2048, 512)],
)
target = out / 'wikitext2-prefix.json'
target.write_text(json.dumps(record, indent=2, ensure_ascii=True) + '\n')
print(json.dumps({k: v for k, v in record.items() if k != 'passages'}, ensure_ascii=True))
