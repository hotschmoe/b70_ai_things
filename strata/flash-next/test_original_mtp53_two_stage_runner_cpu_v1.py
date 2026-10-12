import copy,unittest
import run_original_mtp53_two_stage_v1 as r
class Controls(unittest.TestCase):
 def control(self):
  rows=[];position=1
  for trial,accepted in enumerate((0,1,0)):
   for stage in(0,1):rows.append('MTP53_PAIR_CONTROL_RESTORED trial=%d stage=%d device=%d lb=%d le=%d base_position=%d touched_cells=%d state_KV_exact=1'%(trial,stage,stage,24*stage,24*(stage+1),position,position+2))
   keep=2 if accepted else 1;position+=keep;rows.append('MTP53_PAIR_CONTROL_STEP trial=%d accepted=%d committed=%d position=%d target_greedy_equal=1'%(trial,accepted,keep,position))
  return '\n'.join(rows)+'\n'
 def test_actual_controls_roster(self):
  text=self.control();self.assertTrue(r.control_rows(text)['both_stage_state_restored'])
  with self.assertRaises(ValueError):r.control_rows(text.replace('stage=1 device=1','stage=1 device=0',1))
  with self.assertRaises(ValueError):r.control_rows(text.replace('trial=1 accepted=1','trial=1 accepted=0'))
 def test_actual_metadata_command(self):
  cfg=r.read(r.BASE)
  for mode,ids in(('controls',[248045]),('off',[248045]*2048),('on',[248045]*2048)):
   argv=r.native_options(cfg,mode,ids);self.assertEqual(argv[argv.index('--layer-split')+1],'24');self.assertEqual(argv[argv.index('--prefill')+1],'512');self.assertEqual(argv[argv.index('--max-context')+1],'4096');self.assertEqual(argv[0],'/diagnostic/original-mtp53-two-stage')
  bad=copy.deepcopy(cfg);bad['native_argv'][bad['native_argv'].index('--spec')+1]='1'
  with self.assertRaises(ValueError):r.native_options(bad,'controls',[248045])
 def test_natural_whole_history_and_prompt(self):
  text=''
  position=0
  for n in(512,512,512,511):text+='MTP53_PAIR_PREFILL rows=%d begin=%d end=%d target_batched=1 owned_draft_catchup=0 device=1\n'%(n,position,position+n);position+=n
  for i in range(512):
   if i:text+='MTP53_PAIR_WINDOW input=7 guess=-1 accepted=0 committed=1 position=%d remaining=%d\n'%(2048+i,512-i)
   text+='MTP53_PAIR_TOKEN ordinal=%d id=7 eos=0 host_monotonic_ns=%d\n'%(i+1,i+1)
  text+='MTP53_PAIR_PASS mode=off prompt_tokens=2048 generated=512 useful=512 natural_eos=0 accepted_windows=0 rejected_windows=0 prompt_wall_ms=1 decode_wall_ms=1 draft_capacity=4096 draft_cells=0 resident_original_tensor_control_both_arms=1 fullmath_qualified=0 API_qualified=0 speed_qualified=0\n'
  self.assertEqual(len(r.natural_rows(text,'off')['ids']),512)
  with self.assertRaises(ValueError):r.natural_rows(text.replace('rows=511','rows=510'),'off')
  with self.assertRaises(ValueError):r.natural_rows(text.replace('position=2049','position=2050',1),'off')
if __name__=='__main__':unittest.main()
