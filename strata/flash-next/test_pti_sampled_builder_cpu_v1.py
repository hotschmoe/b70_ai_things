import unittest,copy,subprocess
from unittest.mock import patch
import build_pti_sampled_queries_v1 as b
class Tests(unittest.TestCase):
    def fixture(self):
        cmd=b.recipe.recipe(b.SOURCE,b.OUTPUT,b.PROPOSAL_SHA);cmd[1]='create';identifier='a'*64
        obj={'Id':identifier,'Name':'/'+cmd[cmd.index('--name')+1],'Config':{'Image':b.p.IMAGE,'Entrypoint':['/bin/bash'],'Cmd':cmd[-2:],'User':cmd[cmd.index('--user')+1],'Labels':{'b70.pti.source':b.PROPOSAL_SHA}},'HostConfig':{'Devices':[],'DeviceRequests':[],'Binds':[cmd[i+1]for i,x in enumerate(cmd)if x=='-v'],'NanoCpus':2000000000,'Memory':4<<30,'MemorySwap':4<<30,'PidsLimit':512,'NetworkMode':'bridge'}}
        return cmd,identifier,obj
    def test_full_recipe(self):
        cmd,identifier,obj=self.fixture();b.inspection(obj,cmd,identifier)
        for field in('Image','Cmd','User'):
            changed=copy.deepcopy(obj);changed['Config'][field]='foreign'
            with self.assertRaises(ValueError):b.inspection(changed,cmd,identifier)
    def test_typed_resources(self):
        cmd,identifier,obj=self.fixture();obj['HostConfig']['NanoCpus']=2e9
        with self.assertRaises(ValueError):b.inspection(obj,cmd,identifier)
    def test_ambiguous_create_recovery(self):
        import json
        cmd,identifier,obj=self.fixture()
        with patch.object(b.subprocess,'run',return_value=subprocess.CompletedProcess([],0,json.dumps([obj]),'')):
            self.assertEqual(b.recover_created(cmd)[0],identifier)
        obj['Config']['Labels']['b70.pti.source']='foreign'
        with patch.object(b.subprocess,'run',return_value=subprocess.CompletedProcess([],0,json.dumps([obj]),'')):
            with self.assertRaises(ValueError):b.recover_created(cmd)
    def test_absence_requires_original_error(self):
        cmd,_,_=self.fixture()
        with patch.object(b.subprocess,'run',return_value=subprocess.CompletedProcess([],1,'','daemon unavailable')):
            with self.assertRaises(ValueError):b.recover_created(cmd)
    def test_failure_retains_receipt_source(self):
        import inspect
        source=inspect.getsource(b.main);self.assertIn("errors.append('cleanup: '",source);self.assertIn("write('receipt.json'",source);self.assertIn('original_creator_terminal=creation_terminal',source)
if __name__=='__main__':unittest.main()
