"""Opt-in diagnostics only; unchanged canonical stat permits old hash, never promotion."""
import ast,copy,hashlib,json,os,time,types
from pathlib import Path
from serial37_canonical_json_v3 import canonical
OPTIONS=None
SCOPE='diagnostic_authenticated_prior_whole_hash_current_stat_pages'
def require(ok,msg):
 if not ok:raise ValueError(msg)
def stat_fd(path):
 path=Path(path);require(path.is_absolute() and path.resolve()==path and not path.is_symlink(),'Canonical nonsymlink model path required')
 def fields(s):return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
 before=fields(path.stat());fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
 try:opened=fields(os.fstat(fd));after=fields(path.stat());require(before==opened==after,'Opened model FD/path stat identity changed');return opened
 finally:os.close(fd)
def configure(mode='qualifying',prior=None):
 global OPTIONS
 require(mode in('qualifying','diagnostic'),'Explicit identity mode required');require((prior is not None)==(mode=='diagnostic'),'Diagnostic alone requires authenticated prior');OPTIONS=dict(mode=mode,prior=prior)
def authenticate(c,prior,lock_path,lock,shards):
 # Authentication failure refuses admission. Stat mismatch alone can invoke the
 # original complete scanner; an untrusted receipt cannot authorize a shortcut.
 root=Path(prior['root']);public=Path(prior['public']);require(root.is_absolute() and root.resolve()==root and public.is_absolute() and public.resolve()==public,'Canonical prior root/public required')
 for p,key in [(root/'report.json','report_sha256'),(public,'public_sha256'),(root/'post-full4.json','post_full4_sha256')]:require(not p.is_symlink() and c.sha(p)==prior[key],'Authenticated prior receipt changed '+key)
 report=c.read(root/'report.json');proof=c.read(public);scan=c.read(root/'post-full4.json');require(report['passed'] is True and proof['passed'] is True and proof['producer_report_sha256']==prior['report_sha256'] and Path(proof['root']).resolve()==root,'Prior independently closed owner required')
 require(c.sha(root/'post-full4.json')==report['artifacts']['post-full4.json'] and canonical(scan)==canonical(report['post_full4']),'Prior saved whole-scan association differs')
 expected=[x for x in lock['files'] if x['path'].startswith('UD-Q4_K_XL/')];require(scan['passed'] is True and len(scan['rows'])==len(expected)==len(shards)==4 and scan['lock_sha256']==c.sha(lock_path) and scan['model_revision']==lock['revision'],'Prior exact model publisher identity required');require(scan['read_mode']=='ordinary complete buffered read; no direct IO/cache mutation','Actual completed whole buffered scan required')
 require(report['post_health']['passed'] is True and report['actual_model_terminal_epoch']<=report['post_health']['finished_epoch']<=report['post_before_pages']['epoch']<=scan['started']<=scan['finished']<=report['post_pages']['epoch']<=report['finished_epoch']<=time.time(),'Prior closing chronology differs')
 for row,want,path in zip(scan['rows'],expected,shards):require(row['path']==str(path) and row['sha256']==row['expected_sha256']==want['sha256'] and row['bytes']==want['size'] and row['passed'] is True and row['stat_before']==row['stat_after'],'Prior original publisher row differs')
 return scan

def identity(c,lock_path,lock,shards,output,after,fresh_scan,prior):
 started=time.time();require(started>=after,'Actual current identity boundary required');saved=authenticate(c,prior,lock_path,lock,shards);fallback=None
 try:
  current=[stat_fd(p)for p in shards];require(current==[row['stat_after']for row in saved['rows']],'Current canonical stat5 differs from prior whole scan')
 except (ValueError,OSError) as exc:fallback=type(exc).__name__+': '+str(exc)
 if fallback is None:rows=copy.deepcopy(saved['rows']);fresh=None
 else:
  fresh=fresh_scan(lock_path,lock,shards,Path(output).with_name(Path(output).stem+'-fresh-fallback.json'),after);require(fresh['passed'] is True,'Fresh whole-file fallback failed');rows=copy.deepcopy(fresh['rows']);require(fresh['lock_sha256']==saved['lock_sha256'] and fresh['model_revision']==saved['model_revision'] and len(rows)==len(saved['rows']),'Fresh fallback publisher identity differs');require(all(x['path']==old['path'] and x['sha256']==x['expected_sha256']==old['expected_sha256'] and x['bytes']==old['bytes'] and x['passed'] is True and x['stat_before']==x['stat_after'] for x,old in zip(rows,saved['rows'])),'Fresh fallback publisher rows differ');require([stat_fd(p)for p in shards]==[x['stat_after']for x in rows],'Fresh fallback/current opened FD stat differs')
 value=dict(schema='diagnostic-authenticated-prior-stat-pages-v1',passed=True,lock_sha256=saved['lock_sha256'],model_revision=saved['model_revision'],started=started,finished=time.time(),rows=rows,after_terminal_and_post_health_epoch=after,read_mode='authenticated_prior_whole_hash_current_opened_FD_stat5' if fresh is None else 'fresh_complete_buffered_fallback',fresh_model_bytes_read=fresh is not None,prior=prior,fallback_reason=fallback,fresh_fallback=fresh,diagnostic_identity=True,speed_qualified=False,stability_qualified=False,shelf_qualified=False)
 c.write(output,value);return value

def rename(tree):
 class Names(ast.NodeTransformer):
  def visit_Constant(self,n):
   if isinstance(n.value,str):n.value=n.value.replace('pre_full4','pre_diagnostic_identity').replace('post_full4','post_diagnostic_identity').replace('-full4.json','-diagnostic-identity.json').replace('whole_matched_arms',SCOPE).replace('Fresh complete campaign pre-scan failed','Diagnostic precheck or fresh fallback failed').replace('Fresh complete campaign post-scan failed','Diagnostic postcheck or fresh fallback failed')
   return n
 return Names().visit(tree)
def stats(report,current):
 require(report['identity_scope']==SCOPE,'Explicit diagnostic scope required');a=report['pre_diagnostic_identity'];b=report['post_diagnostic_identity'];baseline={x['path']:x['stat_after']for x in a['rows']};require(a['passed'] is True and b['passed'] is True and current==baseline=={x['path']:x['stat_after']for x in b['rows']},'Actual pre/post diagnostic stat identity differs');require(all(x['pre_stat']==x['post_stat']==baseline for x in report['per_arm'].values()),'Inter-arm mutation refuses diagnostic')
 return True
def ports(namespace,base):
 require(OPTIONS is not None,'Identity lane configured before protected owner construction')
 if OPTIONS['mode']=='qualifying':return base(namespace)
 if 'boundary_base'in base.__globals__:base=base.__globals__['boundary_base'].boundary_ports
 captured=[]
 def compiler(source,*args,**kwargs):
  if isinstance(source,ast.AST) and args and '[campaign-boundary-identity]'in str(args[0]):captured.append(copy.deepcopy(source))
  return compile(source,*args,**kwargs)
 fn=types.FunctionType(base.__code__,{**base.__globals__,'compile':compiler},base.__name__,base.__defaults__,base.__closure__);result=fn(namespace);require(len(captured)==1,'Exact original campaign identity main required');tree=rename(captured[0]);count=0
 for n in ast.walk(tree):
  if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='full_buffered_identity':n.func=ast.Name(id='diagnostic_identity',ctx=ast.Load());count+=1
 require(count==2,'Only actual pre/post whole-scan boundaries substituted');scanner=result['life'].q.upload_v2.full_buffered_identity
 result.update(diagnostic_identity=lambda *args:identity(result['c'],*args,scanner,OPTIONS['prior']),campaign_stats=stats);namespace['campaign_stats']=stats
 main=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='main');attempt=next(n for n in main.body if isinstance(n,ast.Try));at=next(i for i,n in enumerate(attempt.finalbody)if isinstance(n,ast.Expr)and 'report.json'in ast.unparse(n))
 attempt.finalbody[at:at]=ast.parse("report['diagnostic_identity']=True;report['diagnostic_only']=True;report['fresh_model_bytes_read']=any(report.get(k,{}).get('fresh_model_bytes_read',False) for k in ('pre_diagnostic_identity','post_diagnostic_identity'));report['speed_qualified']=False;report['stability_qualified']=False;report['shelf_qualified']=False").body
 exec(compile(ast.fix_missing_locations(tree),__file__+'[explicit-diagnostic-owner]','exec'),result);return result

def reader_ports(active,base):
 captured=[]
 def compiler(source,*args,**kwargs):
  if isinstance(source,ast.AST)and args and '[campaign-reader]'in str(args[0]):captured.append(copy.deepcopy(source))
  return compile(source,*args,**kwargs)
 fn=types.FunctionType(base.__code__,{**base.__globals__,'compile':compiler},base.__name__,base.__defaults__,base.__closure__);result=fn(active);require(len(captured)==1,'Exact protected reader main required');tree=rename(captured[0]);exec(compile(ast.fix_missing_locations(tree),__file__+'[explicit-diagnostic-reader]','exec'),result)
 def recollect(root,report,phase,prefix=''):
  value=result['c'].read(Path(root)/(prefix+phase+'-diagnostic-identity.json'));require(canonical(value)==canonical(report[phase+'_diagnostic_identity']) and value['schema']=='diagnostic-authenticated-prior-stat-pages-v1' and value['passed'] is True and value['diagnostic_identity'] is True and value['speed_qualified'] is value['stability_qualified'] is value['shelf_qualified'] is False,'Actual explicit diagnostic receipt required')
  lock=result['c'].read(active.HERE/'model-lock.json');paths=[Path(x['path'])for x in value['rows']];saved=authenticate(result['c'],value['prior'],active.HERE/'model-lock.json',lock,paths);require([stat_fd(p)for p in paths]==[x['stat_after']for x in value['rows']],'Current opened FD stat differs')
  if value['fresh_model_bytes_read'] is False:require(value['fallback_reason'] is None and value['fresh_fallback'] is None and canonical(saved['rows'])==canonical(value['rows']),'Prior rows cannot be synthesized')
  else:
   fresh=value['fresh_fallback'];p=Path(root)/(prefix+phase+'-diagnostic-identity-fresh-fallback.json');require(canonical(result['c'].read(p))==canonical(fresh) and fresh['passed'] is True and fresh['lock_sha256']==saved['lock_sha256'] and fresh['model_revision']==saved['model_revision'] and len(fresh['rows'])==4 and all(x['path']==old['path'] and x['sha256']==x['expected_sha256']==old['expected_sha256'] and x['bytes']==old['bytes'] and x['stat_before']==x['stat_after'] and x['passed'] is True for x,old in zip(fresh['rows'],saved['rows'])) and canonical(fresh['rows'])==canonical(value['rows']) and fresh['read_mode']=='ordinary complete buffered read; no direct IO/cache mutation','Exact saved complete fallback required')
  result['pages'](report[phase+'_pages'],paths[2]);require(value['started']<=value['finished'],'Diagnostic chronology differs')
  if phase=='pre':result['pages'](report['pre_after_pages'],paths[2]);require(report['pre_pages']['epoch']<=value['started']<=value['finished']<=report['pre_after_pages']['epoch'],'Pre known-page/stat brackets differ')
  else:require(max(report['actual_model_terminal_epoch'],report['post_health']['finished_epoch'])<=value['started']<=value['finished']<=report['post_pages']['epoch'],'Post native/health/page/stat brackets differ')
  return value
 result['full4']=recollect;result['finalized_binding']=result['_finalized_binding'];return result
