import unittest,copy
from pathlib import Path
import build_pti_paused_regular_lists_v1 as b
class Tests(unittest.TestCase):
    def fixture(self):
        cmd=b.recipe.recipe(b.SOURCE,Path('/tmp/b70-paused-builder-never-created-cpu-fixture'),b.PROPOSAL_SHA);identifier='a'*64
        obj={'Id':identifier,'Name':'/'+cmd[cmd.index('--name')+1],'Config':{'Image':b.p.IMAGE,'Entrypoint':['/bin/bash'],'Cmd':cmd[-2:],'User':cmd[cmd.index('--user')+1],'Labels':{'b70.pti.source':b.PROPOSAL_SHA}},'HostConfig':{'Privileged':False,'DeviceCgroupRules':None,'GroupAdd':None,'CapAdd':None,'Devices':[],'DeviceRequests':[],'Binds':[cmd[i+1]for i,x in enumerate(cmd)if x=='-v'],'NanoCpus':2000000000,'Memory':4<<30,'MemorySwap':4<<30,'PidsLimit':512,'NetworkMode':'bridge'}};return cmd,identifier,obj
    def test_normal_recipe(self):
        cmd,identity,obj=self.fixture();b.inspection(obj,cmd,identity)
    def test_foreign_device_domains_rejected(self):
        cmd,identity,obj=self.fixture()
        for key,value in(('Privileged',True),('DeviceCgroupRules',['c *:* rwm']),('GroupAdd',['44']),('CapAdd',['ALL'])):
            bad=copy.deepcopy(obj);bad['HostConfig'][key]=value
            with self.assertRaises(ValueError):b.inspection(bad,cmd,identity)
    def test_owner_resources_types_and_failure_retention_source(self):
        cmd,identity,obj=self.fixture();obj['HostConfig']['Memory']=float(4<<30)
        with self.assertRaises(ValueError):b.inspection(obj,cmd,identity)
        text=Path(b.__file__).read_text();self.assertIn('lease_release_safe=False',text);self.assertIn('while creator.poll()is None:',text);self.assertIn('rm_completed=True',text);self.assertNotIn('proc.terminate()',text)
if __name__=='__main__':unittest.main()
