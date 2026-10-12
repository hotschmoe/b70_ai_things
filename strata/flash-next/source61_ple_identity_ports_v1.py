"""Explicit unqualified identity reuse and GPU-child/CPU-close campaign port."""
import ast,copy,json,sys,time,types,subprocess,os
from pathlib import Path
from serial37_canonical_json_v3 import canonical

OPTIONS=None

def require(ok,msg):
 if not ok:raise ValueError(msg)

def configure(mode,prior=None,prior_sha=None,gpu_child=False):
 global OPTIONS
 require(mode in ('qualifying','diagnostic'),'Explicit identity mode required')
 require((prior is not None and prior_sha is not None) if mode=='diagnostic' else prior is None and prior_sha is None,'Diagnostic requires exact prior whole scan; qualifying forbids reused identity')
 OPTIONS=dict(mode=mode,prior=prior,prior_sha=prior_sha,gpu_child=gpu_child)

def renamed(tree):
 class Rename(ast.NodeTransformer):
  def visit_Constant(self,node):
   if isinstance(node.value,str):
    node.value=node.value.replace('pre_full4','pre_diagnostic_identity').replace('post_full4','post_diagnostic_identity').replace('-full4.json','-diagnostic-identity.json').replace('whole_matched_arms','diagnostic_prior_whole_hash_current_stat_pages')
   return node
 return Rename().visit(tree)

def prior_binding(c,lock_path,lock,shards,prior,prior_sha):
 path=Path(prior)
 require(path.is_absolute() and path.resolve()==path and path.is_file() and not path.is_symlink(),'Canonical prior whole identity receipt required')
 require(c.sha(path)==prior_sha,'Authenticated prior whole identity receipt changed')
 value=c.read(path);expected=[x for x in lock['files'] if x['path'].startswith('UD-Q4_K_XL/')]
 require(value['passed'] is True and value['lock_sha256']==c.sha(lock_path) and value['model_revision']==lock['revision'] and len(value['rows'])==len(expected)==len(shards)==4,'Complete pinned prior publisher identity required')
 require(value['started']<=value['finished']<=time.time() and value['read_mode']=='ordinary complete buffered read; no direct IO/cache mutation','Prior must actually be a completed whole byte scan')
 for row,want,path in zip(value['rows'],expected,shards):
  path=Path(path);require(path.resolve()==path and not path.is_symlink() and path.is_file(),'Canonical current publisher path required')
  require(row['path']==str(path) and row['sha256']==row['expected_sha256']==want['sha256'] and row['bytes']==want['size'] and row['passed'] is True and row['stat_before']==row['stat_after']==c.stat_signature(path),'Prior hash/current canonical stat5 differs')
 require(c.sha(prior)==prior_sha,'Prior receipt changed during current stat census')
 return value

def ports(namespace,base_ports):
 captures=[]
 if 'boundary_base' in base_ports.__globals__:base_ports=base_ports.__globals__['boundary_base'].boundary_ports
 def compiler(source,*args,**kwargs):
  if isinstance(source,ast.AST) and args and '[campaign-boundary-identity]' in str(args[0]):captures.append(copy.deepcopy(source))
  return compile(source,*args,**kwargs)
 fn=types.FunctionType(base_ports.__code__,{**base_ports.__globals__,'compile':compiler},base_ports.__name__,base_ports.__defaults__,base_ports.__closure__)
 result=fn(namespace)
 namespace['campaign_stats']=result['campaign_stats']
 require(len(captures)==1,'Exact existing generated campaign main required')
 tree=captures[0];main=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='main');attempt=next(x for x in main.body if isinstance(x,ast.Try))
 if OPTIONS['mode']=='diagnostic':
  tree=renamed(tree)
  changed=0
  for node in ast.walk(tree):
   if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='full_buffered_identity':node.func=ast.Name(id='diagnostic_identity',ctx=ast.Load());changed+=1
  require(changed==2,'Exactly campaign pre/post complete scan substitutions required')
  def diagnostic_identity(lock_path,lock,shards,output,after):
   started=time.time();require(started>=after,'Current diagnostic identity chronology invalid')
   prior=prior_binding(result['c'],lock_path,lock,shards,OPTIONS['prior'],OPTIONS['prior_sha'])
   value=dict(schema='diagnostic-prior-whole-stat-pages-v1',passed=True,lock_sha256=prior['lock_sha256'],model_revision=prior['model_revision'],started=started,finished=time.time(),rows=copy.deepcopy(prior['rows']),after_terminal_and_post_health_epoch=after,read_mode='prior_authenticated_whole_hash_plus_current_stat5',fresh_model_bytes_read=False,prior_path=OPTIONS['prior'],prior_sha256=OPTIONS['prior_sha'],prior_finished=prior['finished'],speed_qualified=False,shelf_qualified=False)
   result['c'].write(output,value);return value
  def campaign_stats(report,current):
   require(report['identity_scope']=='diagnostic_prior_whole_hash_current_stat_pages','Explicit unqualified identity scope required')
   before=report['pre_diagnostic_identity'];after=report['post_diagnostic_identity'];baseline={x['path']:x['stat_after'] for x in before['rows']}
   require(current==baseline=={x['path']:x['stat_after'] for x in after['rows']} and before['prior_sha256']==after['prior_sha256'],'Diagnostic current/final/prior identity differs')
   require(all(x['pre_stat']==x['post_stat']==baseline for x in report['per_arm'].values()),'Inter-arm diagnostic stat mutation')
   return True
  result.update(diagnostic_identity=diagnostic_identity,campaign_stats=campaign_stats)
  namespace['campaign_stats']=campaign_stats
 if OPTIONS['mode']=='diagnostic':
  at=next(i for i,n in enumerate(attempt.finalbody) if isinstance(n,ast.Expr) and 'report.json' in ast.unparse(n))
  attempt.finalbody[at:at]=ast.parse("report['fresh_model_bytes_read']=False\nreport['diagnostic_only']=True\nreport['clean_performance_qualified']=False\nreport['shelf_qualified']=False").body
 if OPTIONS['gpu_child']:
  # Remove only CPU campaign-close identity from the GPU owner. All per-arm
  # health/journal/page/source/retirement operations and failure health remain.
  final=attempt.finalbody;close=next(n for n in final if isinstance(n,ast.Try) and 'post_before_pages' in ast.unparse(n))
  final[final.index(close)]=ast.parse("report['GPU_close_pending_CPU_identity']=True\nreport['GPU_close_qualified']=False").body[0]
  # Pending qualification must survive failure too; do not run a public reader
  # until the coordinator has sealed the fresh closing identity.
  final.extend(ast.parse("report['GPU_close_pending_CPU_identity']=True\nreport['qualification_pending_CPU_identity']=True").body)
  # These flags need serialization before write(report), not after it.
  final[-2:]=[]
  at=next(i for i,n in enumerate(final) if isinstance(n,ast.Expr) and 'report.json' in ast.unparse(n))
  final[at:at]=ast.parse("report['GPU_close_pending_CPU_identity']=True\nreport['qualification_pending_CPU_identity']=True\nreport['identity_mode']='"+OPTIONS['mode']+"'").body
 if OPTIONS['gpu_child']:
  at=next(i for i,n in enumerate(attempt.finalbody) if isinstance(n,ast.Expr) and 'report.json' in ast.unparse(n))
  attempt.finalbody[at+1:at+1]=ast.parse("safe_GPU_release(report,out)").body
  result['safe_GPU_release']=lambda report,out:safe_GPU_release(result,report,out)
 exec(compile(ast.fix_missing_locations(tree),__file__+'[explicit-identity-mode]','exec'),result)
 return result

def safe_GPU_release(namespace,report,out):
 """Keep the actual lease owner alive on unknown retirement or failed health."""
 errors=[];c=namespace['c'];out=Path(out)
 try:
  health=report['post_health'];label=report['aggregate_post_health_label']
  require(health['passed'] is True and canonical(c.read(out/(label+'-health.json')))==canonical(health),'Actual closing post health failed or changed')
  from native_rms_kernel_journal_v5 import reject_faults
  reject_faults((out/(label+'-journal.log')).read_text())
  require(health['started_epoch']>=report['actual_model_terminal_epoch'],'Closing health before all actual native terminals')
  for phase in report['per_arm']:
   directory=out/phase
   if (directory/'command.json').exists():
    command=c.read(directory/'command.json');name=command[command.index('--name')+1]
    require(c.absent(name),'Actual owned native still exists at GPU close')
    value=c.read(directory/'result.json')
    require(value['removal']['removed'] is True and value['terminal']['State']['Running'] is False,'Typed native removal/terminal required before lease release')
  report['typed_GPU_release_passed']=True
 except BaseException as exc:errors.append(type(exc).__name__+': '+str(exc))
 c.write(out/'GPU-release.json',dict(passed=not errors,errors=errors,checked_epoch=time.time(),producer_pid=report['producer_pid'],producer_start_ticks=report['producer_start_ticks'],post_health_label=report.get('aggregate_post_health_label'),post_health_sha256=c.sha(out/(report['aggregate_post_health_label']+'-health.json')) if report.get('aggregate_post_health_label') else None,CPU_closing_identity_pending=True))
 report['artifacts']={str(p.relative_to(out)):c.sha(p) for p in out.rglob('*') if p.is_file() and p.name!='report.json'}
 if errors:report['errors'].extend('GPU release: '+x for x in errors);report['passed']=False
 c.write(out/'report.json',report)
 if errors:
  # No synthetic EOF, no background GPU owner, and no implicit lease release.
  # Coordinator recovery can interrupt this retained foreground owner.
  while True:time.sleep(1)


def inherited_GPU_lease():
 base=os.environ.get('B70_GPU_LOCK','/mnt/vm_8tb/b70/gpu.lock')
 for fd in Path('/proc/self/fd').iterdir():
  try:
   if os.readlink(fd) in (base+'.0',base+'.1'):return True
  except OSError:pass
 return False

def close_after_child(actor,output,child_rc):
 """CPU coordinator: never execute this inside a GPU lease."""
 c=actor.c;root=Path(output);report=c.read(root/'report.json');
 release=c.read(root/'GPU-release.json');require(release['passed'] is True,'Typed GPU release receipt required')
 require(release['producer_pid']==report['producer_pid'] and release['producer_start_ticks']==report['producer_start_ticks'] and release['post_health_sha256']==c.sha(root/(release['post_health_label']+'-health.json')),'Exact child owner/post-health association required')
 roster={str(p.relative_to(root)):c.sha(p) for p in root.rglob('*') if p.is_file() and p.name!='report.json'}
 require(roster==report['artifacts'],'GPU child current raw/source artifacts changed')
 c.write(root/'GPU-child-report.json',report)
 require(report.get('qualification_pending_CPU_identity') is True,'Typed GPU child pending close receipt required')
 require(child_rc==0 or report['errors'],'Nonzero child must preserve explicit failure')
 # GPU lease is now released by gpu-run, so public qualification cannot rely
 # on borrowed health: retain the actual child per-arm/post health receipts.
 require(report['post_health']['passed'] is True,'Failed post-health cannot use successful CPU close')
 for phase in report['per_arm']:
  directory=root/phase
  if (directory/'result.json').exists():
   value=c.read(directory/'result.json');require(value['removal']['removed'] is True,'Every actual owned native must be retired before CPU closing')
 lock=c.read(actor.HERE/'model-lock.json');shards=[actor.ROOT/lock['destination']/r['path'] for r in lock['files'] if r['path'].startswith('UD-Q4_K_XL/')]
 report['post_before_pages']=actor.NS['intake'].guard(shards,root/'guardian','campaign-post-before')
 after=max(report['actual_model_terminal_epoch'],report['post_health']['finished_epoch'])
 try:
  require(OPTIONS['mode']=='qualifying','Split close currently requires fresh qualifying identity')
  report['post_full4']=actor.NS['life'].q.upload_v2.full_buffered_identity(actor.HERE/'model-lock.json',lock,shards,root/'post-full4.json',after)
  report['post_pages']=actor.NS['intake'].guard(shards,root/'guardian','campaign-post')
  require(report['post_full4']['passed'],'Fresh CPU closing complete model scan failed')
  actor.NS['campaign_stats'](report,actor.NS['stat_snapshot'](shards))
  require(actor.closure()==report['source_plan_sha256'],'Final current source changed')
 except BaseException as exc:report['errors'].append('CPU closing: '+type(exc).__name__+': '+str(exc))
 report.update(qualification_pending_CPU_identity=False,GPU_close_pending_CPU_identity=False,closing_CPU_coordinator_pid=__import__('os').getpid(),closing_GPU_child_return_code=child_rc,passed=not report['errors'],finished_epoch=time.time())
 report['artifacts']={str(p.relative_to(root)):c.sha(p) for p in root.rglob('*') if p.is_file() and p.name!='report.json'};c.write(root/'report.json',report)
 require(report['passed'],'Actual PLE campaign failed; CPU closure retained')

def reader_ports(active,base_reader):
 """Preserve all non-identity public predicates; explicit diagnostic proof only."""
 if OPTIONS['mode']=='qualifying':return base_reader.reader_ports(active)
 trees=[]
 def compiler(source,*args,**kwargs):
  if isinstance(source,ast.AST):trees.append(copy.deepcopy(source))
  return compile(source,*args,**kwargs)
 boundary=base_reader.boundary
 wrapped=types.FunctionType(boundary.boundary_reader_ports.__code__,{**boundary.boundary_reader_ports.__globals__,'compile':compiler},boundary.boundary_reader_ports.__name__,boundary.boundary_reader_ports.__defaults__,boundary.boundary_reader_ports.__closure__)
 proxy=types.SimpleNamespace(**{**vars(boundary),'boundary_reader_ports':wrapped})
 fn=types.FunctionType(base_reader.reader_ports.__code__,{**base_reader.reader_ports.__globals__,'boundary':proxy,'compile':compiler},base_reader.reader_ports.__name__,base_reader.reader_ports.__defaults__,base_reader.reader_ports.__closure__)
 result=fn(active)
 for tree in trees:
  names={x.name for x in tree.body if isinstance(x,ast.FunctionDef)}
  if names & {'_finalized_binding','phase_binding'}:exec(compile(ast.fix_missing_locations(renamed(tree)),__file__+'[diagnostic-public-predicates]','exec'),result)
 def diagnostic_identity(root,report,phase,prefix=''):
  value=result['c'].read(Path(root)/(prefix+phase+'-diagnostic-identity.json'))
  require(canonical(value)==canonical(report[phase+'_diagnostic_identity']) and value['passed'] is True and value['schema']=='diagnostic-prior-whole-stat-pages-v1' and value['fresh_model_bytes_read'] is False and value['speed_qualified'] is False and value['shelf_qualified'] is False,'Explicit unqualified diagnostic identity required')
  lock=result['c'].read(active.HERE/'model-lock.json');paths=[Path(row['path']) for row in value['rows']]
  prior=prior_binding(result['c'],active.HERE/'model-lock.json',lock,paths,value['prior_path'],value['prior_sha256'])
  require(canonical(prior['rows'])==canonical(value['rows']) and value['read_mode']=='prior_authenticated_whole_hash_plus_current_stat5','No current complete-file scan may be fabricated')
  require(value['started']<=value['finished'],'Diagnostic stat census chronology invalid');result['pages'](report[phase+'_pages'],paths[2])
  if phase=='pre':
   result['pages'](report['pre_after_pages'],paths[2]);require(report['pre_pages']['epoch']<=value['started']<=value['finished']<=report['pre_after_pages']['epoch'],'Diagnostic page/stat brackets differ')
  else:require(value['started']>=max(report['actual_model_terminal_epoch'],report['post_health']['finished_epoch']) and value['finished']<=report['post_pages']['epoch'],'Diagnostic post health/page/stat brackets differ')
  return value
 result['full4']=diagnostic_identity
 result['finalized_binding']=result['_finalized_binding']
 return result
