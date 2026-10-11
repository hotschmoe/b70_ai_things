import copy,json,tempfile,unittest
from pathlib import Path
import source41_host_trace_contract_v2 as a
import test_host_critical_path36_cpu_v1 as fixtures
K=fixtures.K
class StrictControls(unittest.TestCase):
 def fixture(self,root,batched=False):
  path,log,request=fixtures.HostTests().fixture(root);rows=[json.loads(line)for line in path.read_text().splitlines()];rows[0]['pid']=1
  lp='LP -1.25 '+' '.join(str(i)+':-1.25'for i in range(20));request['LP']=[lp];request['stdout']=['T 16',lp,request['done']]
  if batched:
   events=[e for e in rows[1:-1]if not(e['dev']==0 and e['pos']in(0,1))]
   for seq,event in enumerate(events):
    event['seq']=seq;event['ns']=100+seq*100
    if event['kind']==K['GraphHit']:event['kind']=K['GraphPreexisting']
   rows=[dict(rows[0],events=len(events)),*events,dict(rows[-1],records=len(events))]
  self.save(path,rows);return path,log,request,rows
 def save(self,path,rows):path.write_text('\n'.join(json.dumps(row)for row in rows)+'\n')
 def collect(self,p,l,r):return a.collect([p],l,'a'*64,[r],[[0,0,48]],1)
 def test_actual_scoped_batched_fixture_positive_old_rejects(self):
  with tempfile.TemporaryDirectory()as t:
   p,l,r,rows=self.fixture(Path(t),True)
   with self.assertRaises(ValueError):a.primitive.original.collect([p],l,'a'*64,[r],[[0,0,48]])
   result=self.collect(p,l,r);self.assertTrue(result['passed']);self.assertEqual(result['native_namespace_pid'],1);self.assertFalse(result['prompt_internal_device_time_observed'])
 def test_missing_lp_or_missing_scopes_never_shrinks_coverage(self):
  for kind in('stdout','raw','scopes'):
   with tempfile.TemporaryDirectory()as t:
    p,l,r,rows=self.fixture(Path(t),True)
    if kind=='stdout':r['stdout']=[line for line in r['stdout']if not line.startswith('LP ')]
    elif kind=='raw':r['LP']=[]
    else:
     events=[e for e in rows[1:-1]if e['kind']not in(K['LPReadback'],K['LPCPU'])]
     for i,e in enumerate(events):e['seq']=i;e['ns']=100+i*100
     self.save(p,[dict(rows[0],events=len(events)),*events,dict(rows[-1],records=len(events))])
    with self.subTest(kind=kind),self.assertRaises(ValueError):self.collect(p,l,r)
 def test_lp_shape_finite_duplicate_order_and_tordinal(self):
  for kind in('nan','duplicate','short','order','token','rawroster'):
   with tempfile.TemporaryDirectory()as t:
    p,l,r,rows=self.fixture(Path(t));parts=r['LP'][0].split()
    if kind=='nan':parts[1]='nan'
    elif kind=='duplicate':parts[-1]=parts[-2]
    elif kind=='short':parts.pop()
    r['LP']=[' '.join(parts)];r['stdout'][1]=r['LP'][0]
    if kind=='order':r['stdout'][0],r['stdout'][1]=r['stdout'][1],r['stdout'][0]
    elif kind=='token':r['stdout'][0]='T 99'
    elif kind=='rawroster':r['LP']=['LP foreign']
    with self.subTest(kind=kind),self.assertRaises((ValueError,IndexError)):self.collect(p,l,r)
 def test_lp_tie_allows_other_top1_id(self):
  with tempfile.TemporaryDirectory()as t:
   p,l,r,rows=self.fixture(Path(t));self.assertTrue(self.collect(p,l,r)['mandatory_LP20_recollected']);self.assertNotEqual(int(r['LP'][0].split()[2].split(':')[0]),r['output_ids'][0])
 def test_typed_schema_event_footer_and_pid_refusals(self):
  for kind in('schema','kind','seq','footer','pid','duplicatekey'):
   with tempfile.TemporaryDirectory()as t:
    p,l,r,rows=self.fixture(Path(t),True)
    if kind=='schema':rows[0]['schema']=36.0
    elif kind=='kind':rows[1]['kind']=True
    elif kind=='seq':rows[1]['seq']=False
    elif kind=='footer':rows[-1]['records']=float(rows[-1]['records'])
    elif kind=='pid':rows[0]['pid']=999
    self.save(p,rows)
    if kind=='duplicatekey':p.write_text(p.read_text().replace('"schema": 36','"schema": 36, "schema": 36',1))
    with self.subTest(kind=kind),self.assertRaises(ValueError):self.collect(p,l,r)
 def test_lp_scope_source_order_refused(self):
  with tempfile.TemporaryDirectory()as t:
   p,l,r,rows=self.fixture(Path(t),True);read=[e for e in rows[1:-1]if e['kind']==K['LPReadback']];cpu=[e for e in rows[1:-1]if e['kind']==K['LPCPU']]
   for e in read:e['kind']=K['LPCPU']
   for e in cpu:e['kind']=K['LPReadback']
   self.save(p,rows)
   with self.assertRaises(ValueError):self.collect(p,l,r)
 def test_native_namespace_direct_exec_positive_and_mutations(self):
  obj={'HostConfig':{'PidMode':'','Init':None},'Config':{'Entrypoint':['/bin/bash']}};cmd=['docker','run','--entrypoint','/bin/bash','image','-c','exec 2>&1; source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; exec /build/strata --serve flags'];self.assertEqual(a.native_namespace(obj,cmd),1)
  for key,value in [('PidMode','host'),('Init',True),('Init',0)]:
   bad=copy.deepcopy(obj);bad['HostConfig'][key]=value
   with self.assertRaises(ValueError):a.native_namespace(bad,cmd)
if __name__=='__main__':unittest.main()
