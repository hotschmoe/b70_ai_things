import copy,unittest
import run_original_mtp53_artifact_preflight_v1 as r
class Metadata(unittest.TestCase):
 def fixture(self):
  cmd=['docker','run','--name','owned','--label','key=owner','--user','1000:1000','-v','/a:/artifact:ro','-v','/b:/src:ro','--entrypoint','/bin/bash',r.IMAGE,'-lc','script']
  row={'Name':'/owned','Config':{'Labels':{'key':'owner'},'Image':r.IMAGE,'Entrypoint':['/bin/bash'],'Cmd':['-lc','script'],'User':'1000:1000'},'HostConfig':{'Binds':['/b:/src:ro','/a:/artifact:ro'],'Privileged':False,'NetworkMode':'none','Devices':[],'DeviceRequests':[],'DeviceCgroupRules':None,'GroupAdd':None,'CapAdd':None,'NanoCpus':2000000000,'Memory':2<<30,'MemorySwap':2<<30},'RestartCount':0}
  return cmd,row
 def test_reordered_exact_binds_and_unsafe_domains(self):
  cmd,row=self.fixture();r.validate_terminal(row,cmd)
  for key,value in [('Privileged',True),('Devices',[{}]),('GroupAdd',['44']),('Memory',float(2<<30)),('Binds',['/a:/artifact:ro','/a:/artifact:ro'])]:
   bad=copy.deepcopy(row);bad['HostConfig'][key]=value
   with self.assertRaises(ValueError):r.validate_terminal(bad,cmd)
 def test_named_case_variable_absence(self):
  self.assertTrue(r.absent(b'error: no such object: owned\n','owned'));self.assertFalse(r.absent(b'Error: No such object: foreign\n','owned'))
