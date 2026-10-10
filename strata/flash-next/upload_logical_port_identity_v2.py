"""Original producer path provenance for separate admission-only consumer ports."""
from pathlib import Path
HERE=Path(__file__).resolve().parent

def original_producer_file(name):
 from prepare_upload_logical_gate_ports_v2 import MAP
 if name not in MAP:raise ValueError('Undeclared original producer identity')
 return str(HERE/(name+'.py'))
def original_module_file(name,alias):
 if type(alias)is not str or not alias:raise ValueError('Exact original module alias required')
 return original_producer_file(name)
