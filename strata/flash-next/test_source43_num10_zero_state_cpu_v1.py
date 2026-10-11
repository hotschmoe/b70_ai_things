"""Tiny owned zero-before-state acceptance/refusal; no model/source43 execution."""
import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import source43_num10_zero_case_v1 as c
class Controls(unittest.TestCase):
 def check(self,mutation):
  with tempfile.TemporaryDirectory()as t:
   root=Path(t);fields=[]
   for name,size in [('gdn_state_before',3145728),('gdn_conv_before',122880)]:
    path=root/(name+'.raw');data=bytearray(size)
    if name==mutation:data[-1]=1
    path.write_bytes(data);fields.append({'name':name,'path':str(path),'bytes':size})
   record={'frame':{'schema':1,'pid':1,'request':1,'stage':0,'layer':0,'rows':1,'position':0,'token':248045,'reused':0,'completed_nonce':(1<<32)|1,'graph_key':2},'observed':fields}
   with patch('layer0_numerical_qualification_v10.collect_frame',return_value=record):
    if mutation is None:self.assertEqual(c.collect({'ids':[248045],'fresh':1,'pin':None},root,'a'*64),record)
    else:
     with self.assertRaises(ValueError):c.collect({'ids':[248045],'fresh':1,'pin':None},root,'a'*64)
 def test_zero_recurrent_and_conv_are_accepted(self):self.check(None)
 def test_one_nonzero_recurrent_byte_refuses(self):self.check('gdn_state_before')
 def test_one_nonzero_conv_byte_refuses(self):self.check('gdn_conv_before')
if __name__=='__main__':unittest.main()
