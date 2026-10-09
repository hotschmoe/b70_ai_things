#!/usr/bin/env python3
"""Synthetic SHA mapping controls, not native packet/model qualification."""
import copy,hashlib,unittest
from audit_owned_layer0_packet_hashes_v1 import compare_events

class HashTests(unittest.TestCase):
 def setUp(self):
  self.ids=list(range(10));self.fields={name:bytes([i+1])*size for i,(name,size) in enumerate([('attn_input_q81',2880),('attn_mixed',10240),('gdn_output_q81',6912),('gdn_output_gated',24576),('ffn_input_q81',2880),('ffn_mixed',10240),('shared_hq81',720),('shared_hidden_DERIVED',2560),('expert_hq81',7200),('expert_hidden_DERIVED',25600)])};self.events=[]
  def add(role,expert,packet,inputfield,rank=None):
   p=self.fields[packet];x=self.fields[inputfield]
   if rank is not None:p=p[rank*720:(rank+1)*720];x=x[rank*2560:(rank+1)*2560]
   self.events.append(dict(role='blk.0.'+role+'.weight',expert=expert,packet_sha256=hashlib.sha256(p).hexdigest(),input_sha256=hashlib.sha256(x).hexdigest(),stored_s_used=False))
  for role,p,x in [('attn_qkv','attn_input_q81','attn_mixed'),('attn_gate','attn_input_q81','attn_mixed'),('ssm_out','gdn_output_q81','gdn_output_gated'),('ffn_gate_shexp','ffn_input_q81','ffn_mixed'),('ffn_up_shexp','ffn_input_q81','ffn_mixed'),('ffn_down_shexp','shared_hq81','shared_hidden_DERIVED')]:add(role,None,p,x)
  for rank,expert in enumerate(self.ids):
   for role in ('ffn_gate_exps','ffn_up_exps'):add(role,expert,'ffn_input_q81','ffn_mixed')
   add('ffn_down_exps',expert,'expert_hq81','expert_hidden_DERIVED',rank)
 def test_complete_consumer_mapping(self):
  rows=compare_events(self.events,self.fields,self.ids);self.assertEqual(len(rows),36);self.assertTrue(all(row['packet_bytes_equal'] and row['input_F32_bytes_equal'] for row in rows))
 def test_input_bits_change_packet_hash_stable_distinguished(self):
  self.events[0]['input_sha256']='0'*64;row=compare_events(self.events,self.fields,self.ids)[0];self.assertFalse(row['input_F32_bytes_equal']);self.assertTrue(row['packet_bytes_equal']);self.assertFalse(row['packet_byte_diff_positions_available'])
 def test_packet_mismatch_reported_without_numeric_pass(self):
  self.events[0]['packet_sha256']='0'*64;row=compare_events(self.events,self.fields,self.ids)[0];self.assertFalse(row['packet_bytes_equal']);self.assertNotIn('passed',row)
 def test_missing_duplicates_rank_and_hash_negative(self):
  for change in ('missing','duplicate','rank','badSHA','storedS'):
   events=copy.deepcopy(self.events);ids=self.ids[:]
   if change=='missing':events.pop()
   elif change=='duplicate':events[0]=events[1].copy()
   elif change=='rank':ids[0],ids[1]=ids[1],ids[0]
   elif change=='badSHA':events[0]['input_sha256']='bad'
   else:events[0]['stored_s_used']=True
   with self.assertRaises(ValueError):compare_events(events,self.fields,ids)
 def test_derived_hidden_scope_is_explicit(self):
  rows=compare_events(self.events,self.fields,self.ids);hq=[r for r in rows if 'hq81' in r['native_packet_field']];self.assertEqual(len(hq),11);self.assertTrue(all('NOTobserved' in r['native_input_provenance'] for r in hq))

if __name__=='__main__':unittest.main()
