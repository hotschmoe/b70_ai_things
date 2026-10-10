"""Actual native markers, Docker timestamps and original pre-source receipts."""
import datetime,re
from pathlib import Path
import native_rms_publisher_binding_v2 as p
def native_trace(path):
 lines=Path(path).read_text().splitlines()
 markers=[line for line in lines if line.startswith('RMS37_')]
 p.require(len(markers)==5,'Exact unique native device/frame/result marker roster required')
 device=re.fullmatch(r'RMS37_DEVICE pid=([1-9][0-9]*) backend=level_zero affinity=0 name=(.+) vendor=(.+) driver=(.+)',markers[0]);p.require(device is not None,'Actual native GPU device text marker differs')
 for route,line in enumerate(markers[1:4]):p.require(line==f'RMS37_FRAME route={route} graph_replay={route} actual_compiled_hc=1 fields=7 synthetic_norm=1 synthetic_zero_down_up=1 original_model_math_qualified=0','Actual exact direct/replay frame order differs')
 p.require(markers[4]=='RMS37_RESULT routes=3 owned_allocations_freed=1 graph_retired=1 device_intrinsics_qualified=0 full_model_math_qualified=0','Actual graph/free/result marker differs')
 return {'actual_device_text':markers[0],'actual_leaf_pid':int(device.group(1)),'actual_unique_frame_free_markers':True,'PCI_identity_inferred':False,'device_intrinsics_qualified':False}
def docker_epochs(report):
 times=[]
 for kind,commandkey in [('compile','compile'),('run','run')]:
  state=report[kind+'_terminal']['State'];command=report[commandkey+'_command']
  start=datetime.datetime.fromisoformat(state['StartedAt'].replace('Z','+00:00'));finish=datetime.datetime.fromisoformat(state['FinishedAt'].replace('Z','+00:00'));p.require(start.tzinfo is not None and finish.tzinfo is not None,'Original Docker timestamp timezone missing')
  times.extend([command['started_command_epoch'],start.timestamp(),finish.timestamp(),command['finished_epoch']])
 times.append(report['GPU_terminal_epoch']);p.epochs(times);p.require(all(a<=b for a,b in zip(times,times[1:])),'Actual compile/container/runtime/EOF/GPU chronology differs');return {'actual_Docker_timestamp_EOF_order_qualified':True}
def pre_publisher(root,report,lock,lock_sha,repository_root):
 from serial37_canonical_json_v3 import matches_saved
 root=Path(root).resolve();proof=report['pre_full4'];p.require(matches_saved(proof,root/'pre-full4.json'),'Original pre-full4 receipt contradicts report')
 expected=[r for r in lock['files'] if r['path'].startswith('UD-Q4_K_XL/')];paths=[Path(repository_root)/lock['destination']/r['path'] for r in expected];p.require(len(expected)==len(proof['rows'])==4 and proof['passed'] is True and proof['lock_sha256']==lock_sha and proof['model_revision']==lock['revision'],'Original pre publisher4 roster/lock/revision differs')
 for row,want,path in zip(proof['rows'],expected,paths):p.require(row['path']==str(path) and row['passed'] is True and row['bytes']==want['size'] and row['sha256']==row['expected_sha256']==want['sha256'] and row['stat_before']==row['stat_after']==p.stat(path),'Original pre publisher byte/hash/path/stat differs')
 before=report['pre_pages'];after=report['pre_full4_after_pages'];health_start=report['pre_health']['rows'][0]['started_command_epoch'];times=[report['started_epoch'],before['epoch'],proof['after_terminal_and_post_health_epoch'],proof['started'],proof['finished'],after['epoch'],health_start];p.epochs(times);p.require(all(a<=b for a,b in zip(times,times[1:])),'Original pre pages/full4/health chronology differs')
 for guard in (before,after):
  p.require(guard['passed'] is True and guard['path']==str(paths[2]) and guard['stat_before']==guard['stat_after']==proof['rows'][2]['stat_after'] and [(r['offset'],r['expected_sha256']) for r in guard['rows']]==list(p.KNOWN_PAGES),'Original pre knownpage roster/path/stat differs')
  for row in guard['rows']:
   saved=Path(row['preserved_path']);p.require(not saved.is_symlink() and saved.resolve().parent==root and row['passed'] is True and row['bytes']==saved.stat().st_size==4096 and p.sha(saved)==row['sha256']==row['expected_sha256'],'Original pre preserved page differs')
 return {'original_pre_publisher4_pages_before_health_qualified':True}
