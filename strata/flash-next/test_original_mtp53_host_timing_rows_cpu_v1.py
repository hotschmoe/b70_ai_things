import unittest
import original_mtp53_host_timing_intake_v1 as p
class Controls(unittest.TestCase):
 def test_complete_synthetic_rows_and_actual_namespace_negative(self):
  records=[]
  def row(scope,name,b=1000,e=2000,owner='(nil)',d=-1,lb=-1,le=-1,T=0,pos=-1):
   records.append('MTP53_HOST_PHASE ordinal=%d scope=%s phase=%s wall_ms=%.9f begin_ns=%d end_ns=%d pid=1 tid=1 owner=%s device=%d lb=%d le=%d T=%d position=%d success=1'%(len(records)+1,scope,name,(e-b)/1e6,b,e,owner,d,lb,le,T,pos))
  for phase in('draft_prepare_submit','draft_record_wait','draft_settle_wait'):
   for _ in range(2048):row('teacher_prompt',phase)
   row('decode',phase)
  for phase in('target_run','target_commit','target_wait_commit'):row('teacher_prompt',phase);row('decode',phase)
  for phase in('own_head_read_argmax','draft_prior_drain','final_target_wait_commit'):row('decode',phase)
  start=0
  for end in(24,1536,2041,2047):row('teacher_prompt','target_prefill_part_run',d=-1,lb=0,le=48,T=end-start,pos=start);start=end
  for T,pos,scope in((1,2047,'teacher_prompt'),(2,2048,'decode')):
   row(scope,'target_stage_run',100,500,'0x1',0,0,24,T,pos);row(scope,'target_stage_submit',110,120,'0x1',0,0,24,T,pos)
   row(scope,'target_stage_run',200,400,'0x2',1,24,48,T,pos);row(scope,'target_stage_submit',210,220,'0x2',1,24,48,T,pos)
  footer='MTP53_HOST_PHASE_SCOPE rows=%d capacity=32768 additional_GPU_waits=0 additional_GPU_queries=0 additional_GPU_copies=0 stage_spans_nested=1 spans_additive=0 host_only=1 production_speed_qualified=0'%len(records)
  text='\n'.join(records+[footer])+'\n';actual={'windows':[{'remaining':'511','position':'2049','committed':'1'}]}
  self.assertGreater(p.collect(text,actual)['row_count'],2048)
  with self.assertRaises(ValueError):p.collect(text.replace('pid=1','pid=2',1),actual)
if __name__=='__main__':unittest.main()
