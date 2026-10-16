"""Future CPU preflight only: verify the 36 retained S39 snapshots before GPU."""
from pathlib import Path
import hashlib,json
P=Path(__file__).resolve().parent
x=json.loads((P/'s39-anchors.json').read_text())
assert x['count']==len(x['anchors'])==36 and x['snapshot_bytes']==747776
for name,expected in x['anchors'].items():
 p=Path(x['anchor_directory'])/name
 assert p.stat().st_size==747776,name
 with p.open('rb')as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
 assert actual==expected,name
print('PASS_36_EXACT_HISTORICAL_S39_ANCHORS_BEFORE_GPU')
