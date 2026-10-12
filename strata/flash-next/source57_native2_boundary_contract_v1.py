"""Matched actual57 OFF/ON scheduler proof; immutable accepted52 history reference."""
import hashlib,json
from pathlib import Path
from source52_native2_fairness_contract_v1 import requests as original_requests
from native2_fairness_protocol_v1 import require,fields,command_for
REFERENCE=Path('/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/source52-native2-fairness-v1-public-v1.json')
REFERENCE_SHA='bf91278a04a78b7c53934c7a11b0a0337d30d6b5ca94038f25785602d5ba0dd2'
def reference():
 before=REFERENCE.stat();raw=REFERENCE.read_bytes();after=REFERENCE.stat();require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns) and hashlib.sha256(raw).hexdigest()==REFERENCE_SHA,'Immutable accepted52 public history reference changed')
 value=json.loads(raw);require(value['passed'] is True and value['source_generation']==52 and value['bounded_native2_default_scheduler_coherence_passed'] is True and value['normal_scheduler_comparison']['full_histories_match_serial'] is True,'Authentic independently admitted52 history reference required')
 return value['normal_scheduler_comparison']['serial_stream_timings']
def requests(rows,expected):
 value=original_requests(rows,expected);require(value['default_scheduler_observed'] is True and value['actual_two_row_events'],'Each actual57 arm must execute normal-scheduler overlap')
 for row,prior in zip(value['streams'],reference()):require(all(row[k]==prior[k] for k in ('role','ids','output_ids','finish')),'Actual57 complete history differs from admitted52 serial reference')
 value['accepted52_history_reference_sha256']=REFERENCE_SHA;return value
def equivalence(off,on):
 require(off['default_scheduler_observed'] is True and on['default_scheduler_observed'] is True and off['actual_two_row_events'] and on['actual_two_row_events'],'Both actual57 OFF/ON owned overlap arms required')
 for a,b in zip(off['streams'],on['streams']):require(all(a[k]==b[k] for k in ('role','ids','output_ids','finish')),'Actual57 OFF/ON whole histories differ')
 return dict(bounded57_boundary_scheduler_coherence_passed=True,actual57_OFF_ON_full_histories_equal=True,accepted52_serial_histories_equal=True,accepted52_history_reference_sha256=REFERENCE_SHA,default_decode_share=.5,same_host_monotonic_domain=True,OFF=dict(stream_timings=off['streams'],arrival_window_gaps=off['arrival_window_gaps'],actual_two_row_events=len(off['actual_two_row_events'])),ON=dict(stream_timings=on['streams'],arrival_window_gaps=on['arrival_window_gaps'],actual_two_row_events=len(on['actual_two_row_events'])),raw_capture_armed=False,HTTP_latency_qualified=False,fullmath_qualified=False,fairness_stability_qualified=False,speed_promotion_qualified=False,scope='actual57 defaultOFF/newON same inputs/source/ABI/chunk/cuts; local token gaps/TTFT, complete histories; one bounded experiment')
