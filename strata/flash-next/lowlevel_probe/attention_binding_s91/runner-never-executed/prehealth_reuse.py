#!/usr/bin/env python3
"""Root revalidates no intervening GPU/stack change; no preflight GPU touch."""
from pathlib import Path
import hashlib,json,os,sys
P=Path(__file__).resolve().parent;OUT=Path(sys.argv[1]);v=json.loads((P/'prehealth-sources.json').read_text())
assert os.environ.get('B70_S91_REVALIDATED_S90_PREHEALTH')=='1','Root must revalidate immediate no-intervening-GPU/stack condition before launch'
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
h=Path(v['post_health']);log=Path(v['card_log']);assert sha(h)==v['post_health_sha256']and sha(log)==v['card_log_sha256']
x=json.loads(h.read_text());assert x['card']['exit']==x['collective']['exit']==0
assert '  card 0: OK'in log.read_text()and'xpu-health: HEALTHY (cards 0 1)'in log.read_text()
for f,n in [(h,'used-prehealth.json'),(log,'used-prehealth-card.log'),(P/'prehealth-sources.json','used-prehealth-sources.json')]:
 with(OUT/n).open('xb')as out:out.write(f.read_bytes());out.flush()
 assert sha(OUT/n)==sha(f)
print('PASS S90 closed health hashes; root revalidated reuse; no pre-leaf GPU health')
