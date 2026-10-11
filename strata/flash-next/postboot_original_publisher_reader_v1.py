"""Recollect original publisher/history with explicit st_dev-only association."""
from pathlib import Path
from serial37_canonical_json_v3 import matches_saved
import native_rms_publisher_binding_v2 as old
from postboot_original_model_association_v1 import historical_stat,require,sha

def publisher(root,report,association,lock,repository_root,stage):
 root=Path(root).resolve();key=stage+'_full4';proof=report[key];require(matches_saved(proof,root/(stage+'-full4.json')),'Original publisher receipt/report join differs');expected=[r for r in lock['files']if r['path'].startswith('UD-Q4_K_XL/')];require(proof['passed']is True and proof['lock_sha256']==association['current_lock_sha256']and proof['model_revision']==lock['revision']and len(proof['rows'])==4,'Original complete ordered publisher lock/revision differs')
 for row,want in zip(proof['rows'],expected):
  path=Path(repository_root)/lock['destination']/want['path'];require(row['path']==str(path)and row['passed']is True and row['bytes']==want['size']and row['sha256']==row['expected_sha256']==want['sha256']and row['stat_before']==row['stat_after'],'Original publisher bytes/hash/stable stat differ');historical_stat(association,path,row['stat_after'])
 old.epochs([proof['started'],proof['finished'],report['started_epoch'],report['GPU_terminal_epoch'],report['post_health']['finished_epoch'],report['post_journal']['finished_epoch'],report['finished_epoch']])
 if stage=='post':
  boundary=max(report['GPU_terminal_epoch'],report['post_health']['finished_epoch'],report['post_journal']['finished_epoch']);require(proof['after_terminal_and_post_health_epoch']==boundary and boundary<=proof['started']<=proof['finished']<=report['finished_epoch'],'Original post publisher terminal/health/journal chronology differs');keys=('before_post_full4_pages','post_pages')
 else:
  require(report['started_epoch']<=report['pre_pages']['epoch']<=proof['after_terminal_and_post_health_epoch']<=proof['started']<=proof['finished']<=report['pre_full4_after_pages']['epoch']<=report['pre_health']['rows'][0]['started_command_epoch'],'Original pre publisher/pages/health chronology differs');keys=('pre_pages','pre_full4_after_pages')
 for name in keys:
  guard=report[name];require(guard['passed']is True and guard['path']==proof['rows'][2]['path']and guard['stat_before']==guard['stat_after']==proof['rows'][2]['stat_after']and[(r['offset'],r['expected_sha256'])for r in guard['rows']]==list(old.KNOWN_PAGES),'Original preserved knownpage source/roster/stat differs')
  for row in guard['rows']:
   path=Path(row['preserved_path']);require(not path.is_symlink()and path.resolve().parent==root and row['passed']is True and row['bytes']==path.stat().st_size==4096 and sha(path)==row['sha256']==row['expected_sha256'],'Original preserved knownpage bytes changed')
 return {'original_publisher_history_recollected':True,'stage':stage,'old_current_stat_gate_passed':False,'historical_health_is_current_health':False,'fresh_model_hash_association':association['current_identity_sha256']}
