import unittest
from probe_serial_control import assess

class Gates(unittest.TestCase):
 def test_semantic_and_cache_independent(self):
  for content,hit,semantic,cache in [('[1,2]',0,True,False),('[1,2,',0,False,False),('[1,2,',1600,False,True),('[1,2]',1600,True,True)]:
   r={'choices':[{'message':{'content':content},'finish_reason':'stop'}],'usage':{'prompt_tokens':4045,'completion_tokens':20,'prompt_tokens_details':{'cached_tokens':hit}}};x=assess(r,[1,2],1);self.assertEqual((x['semantic_passed'],x['cache_passed']),(semantic,cache))
if __name__=='__main__':unittest.main()
