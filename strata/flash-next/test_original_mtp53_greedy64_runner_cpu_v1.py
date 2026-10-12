import unittest
import run_original_mtp53_greedy64_v1 as r
class ActualSequence(unittest.TestCase):
 def text(self,mode,ids):
  lines=[];position=54;emitted=1;windows=0;accepted=tail=0
  for i,token in enumerate(ids):lines.append('MTP53_GREEDY_TOKEN ordinal=%d id=%d eos=%d role=x host_monotonic_ns=%d'%(i+1,token,int(token in(248044,248046)),100+i))
  while emitted<len(ids):
   remaining=64-emitted;keep=2 if mode=='original_mtp_on'and remaining>=2 and ids[emitted]not in(248044,248046)and emitted+2<=len(ids)else 1;a=int(keep==2);windows+=1;position+=keep
   if mode=='original_mtp_on':
    if remaining==1:tail+=1
    elif a:accepted+=1
   lines.append('MTP53_GREEDY_WINDOW window=%d input=%d guess=%d accepted=%d committed=%d position=%d remaining_before=%d'%(windows,ids[emitted-1],ids[emitted],a,keep,position,remaining));emitted+=keep
  lines.append('MTP53_GREEDY_PASS mode=%s prompt_tokens=54 generated=%d useful=%d natural_eos=%d windows=%d accepted_windows=%d rejected_windows=0 single_tail_windows=%d draft_cells=%d budget=64'%(mode,len(ids),len(ids)-int(ids[-1]in(248044,248046)),int(ids[-1]in(248044,248046)),windows,accepted,tail,position if mode=='original_mtp_on'else 0))
  return '\n'.join(lines)
 def test_independent_full64_and_commit_positions(self):
  ids=list(range(100,164));off=r.controls(self.text('target_off',ids),54);on=r.controls(self.text('original_mtp_on',ids),54);self.assertEqual(off['ids'],on['ids']);self.assertEqual(len(on['windows']),32)
  with self.assertRaises(ValueError):r.controls(self.text('original_mtp_on',ids).replace('committed=2','committed=1',1),54)
 def test_eos_is_terminal_without_padding(self):
  text=self.text('target_off',[100,248046]);self.assertTrue(r.controls(text,54)['natural_eos'])
  with self.assertRaises(ValueError):r.controls(text.replace('id=248046','id=101'),54)
