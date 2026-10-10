"""Original detached-launch CLI and session proof; no absence from timeout."""
from pathlib import Path
import json
from full_cache_shared_history_v2 import require

def binding(root,child,plan):
 from full_cache_shared_memory_capture_v6 import recipe_binding
 root=Path(root);launch=child['owned_launch_client'];saved=json.loads((root/'owned-launch-client/receipt.json').read_bytes());command=json.loads((root/'launch.command.json').read_bytes());terminal=json.loads((root/'owned-terminal-inspection.json').read_bytes());desc=launch['launch_descendants']
 require(saved==launch and launch['command']==command and type(launch['return_code'])is int and launch['return_code']==0 and launch['error'] is None and launch['stop_signals']==[] and launch['subreaper']=={'subreaper':True,'owner_pid':child['producer_pid']},'Original Docker creation/ownership/normal result differs')
 require(type(launch['client_pid'])is int and launch['client_pid']>0 and desc['producer_pid']==desc['producer_session']==launch['client_pid'] and desc['launch_session_empty'] is True and desc['tracking_errors']==[] and desc['complete_process_ancestry_from_polling_claimed'] is False and desc['subreaper_wait_and_session_census_required'] is True and launch['Docker_daemon_async_creation_time_bound_claimed'] is False,'Original launch/session cannot be inferred retired')
 require(all(r['actually_reaped'] is True for r in desc['adopted_children'].values()),'Actual adopted Docker launch child remains unretired')
 require((root/'owned-launch-client/stdout.log').read_text().strip()==terminal['Id'] and not (root/'owned-launch-client/stderr.log').read_text().strip(),'Original Docker creation ID or stderr differs')
 require(child['started_epoch']<=launch['client_started_epoch']<=launch['client_terminal_epoch']<=child['finished_epoch'],'Original actual creation/EOF chronology differs')
 recipe_binding(terminal,command,plan['image']);return {'actual_original_creation_ID_joined':True,'actual_owned_launch_session_empty':True,'Docker_daemon_async_creation_time_bound_claimed':False,'full_cache_runtime_qualified':False}
