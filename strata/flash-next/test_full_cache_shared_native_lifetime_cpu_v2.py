import copy,json,unittest
import test_api_owned_terminal_cpu_v2 as fixture
from full_cache_shared_native_lifetime_v2 import recollect

class Lifetime(unittest.TestCase):
 def rows(self):
  original=fixture.Controls().rows();out=[];counter=0
  def meta(phase,generated,window=0,token=-1,finish='',continuation=-1):
   nonlocal counter
   counter+=1;out.append({'kind':'native_receive','engine_pid':77,'line':'FC39 '+json.dumps({'kind':'native_lifetime','pid':77,'event':counter,'rid':1,'slotgen':1,'slot':0,'phase':phase,'generated':generated,'batch_window':window,'token':token,'finish':finish,'continuation':continuation,'actual_HTTP_client_terminal_qualified':False})})
  for row in original:
   if row.get('line','').startswith('BT '):meta('BSTEP_before',1,1)
   out.append(copy.deepcopy(row));line=row.get('line','')
   if line.startswith('DONE '):meta('DONE_emitted',1,finish='length')
   if line.startswith('BADM '):meta('BADM_emitted',1,finish='length',continuation=1)
   if line.startswith('BT '):meta('BT_emitted',2,1,6)
   if line.startswith('BDONE '):meta('BDONE_emitted',2,1,6,'stop',0)
  for sequence,row in enumerate(out,1):row['sequence']=sequence
  return out
 def test_actual_metadata_after_protocol_terminal(self):
  rows=self.rows();r=recollect(rows);self.assertEqual(r['actual_batch_legs'][0]['generated'],2);self.assertFalse(r['full_cache_runtime_qualified'])
  # Frontend end can be observed before the background native reader receives
  # the final already-written FC39 line. It must still be included and joined.
  index=next(i for i,r in enumerate(rows) if r.get('line','').startswith('FC39 ') and 'BDONE_emitted' in r['line']);metadata=rows.pop(index);rows.append(metadata)
  for sequence,row in enumerate(rows,1):row['sequence']=sequence
  self.assertTrue(recollect(rows)['native_batch_metadata_joined'])
 def test_changed_metadata_token_owner_count_or_missing_refused(self):
  for key,value in [('token',7),('pid',78),('generated',3),('slotgen',2),('actual_HTTP_client_terminal_qualified',True)]:
   rows=self.rows();event=next(r for r in rows if r.get('line','').startswith('FC39 ') and 'BT_emitted' in r['line']);raw=json.loads(event['line'][5:]);raw[key]=value;event['line']='FC39 '+json.dumps(raw)
   with self.subTest(key=key),self.assertRaises(ValueError):recollect(rows)
  rows=self.rows();rows=[r for r in rows if 'BDONE_emitted' not in r.get('line','')]
  for sequence,row in enumerate(rows,1):row['sequence']=sequence
  with self.assertRaises(ValueError):recollect(rows)
 def test_multiple_BT_in_one_actual_window(self):
  rows=self.rows();index=next(i for i,r in enumerate(rows) if r.get('line','').startswith('BDONE '));sample=json.loads(next(r['line'][5:] for r in rows if 'BT_emitted' in r.get('line','')))
  sample.update(generated=3,token=7)
  rows[index:index]=[{'kind':'native_receive','engine_pid':77,'line':'BT 0 7 rid=1 slotgen=1'},{'kind':'native_receive','engine_pid':77,'line':'FC39 '+json.dumps(sample)}]
  for row in rows:
   if row.get('line','').startswith('BDONE '):row['line']='BDONE 0 3 stop 1 rid=1 slotgen=1'
   if 'BDONE_emitted' in row.get('line',''):
    raw=json.loads(row['line'][5:]);raw.update(generated=3,token=7);row['line']='FC39 '+json.dumps(raw)
   if row['kind']=='engine_end':row.update(generated_ids=[5,6,7],pinned_eos_ids=[7])
  event=0
  for sequence,row in enumerate(rows,1):
   row['sequence']=sequence
   if row.get('line','').startswith('FC39 '):event+=1;raw=json.loads(row['line'][5:]);raw['event']=event;row['line']='FC39 '+json.dumps(raw)
  self.assertEqual(recollect(rows)['actual_batch_legs'][0]['generated'],3)

if __name__=='__main__':unittest.main()
