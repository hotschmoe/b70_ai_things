"""Read-only raw request derivation; preserve records and compare complete objects."""
import copy,json
from pathlib import Path

def recollect_request(path,saved,capture_dir,stage_ranges,extract):
 original=json.loads(Path(path).read_text());derived=copy.deepcopy(original)
 fresh=extract(derived,capture_dir,True,True,stage_ranges)
 if json.dumps(derived,sort_keys=True)!=json.dumps(saved['raw'],sort_keys=True):raise ValueError('Complete derived raw request differs from saved request')
 if json.dumps(fresh,sort_keys=True)!=json.dumps(saved['meta'],sort_keys=True):raise ValueError('Complete recollected request metadata differs')
 return derived,fresh
