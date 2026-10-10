"""CPU source/metadata controls; never executes a compiled helper."""
import copy,json,unittest
from pathlib import Path
import hc35_host_bulk_fma_v1 as bulk
import qualify_hc35_host_bulk_runtime_v1 as q
class Tests(unittest.TestCase):
 def receipt(self):return dict(schema=2,op='matrixdot',K=32,M=2,mode='shared',output_floats=2,rounding='FE_TONEAREST',flush_to_zero=False,denormals_are_zero=False,device_intrinsics_qualified=False,model_math_qualified=False)
 def test_valid(self):bulk.contract('matrixdot',32,2,'shared',bytes(256),bytes(128));bulk.validate_receipt(self.receipt(),'matrixdot',32,2,'shared',8)
 def test_shape(self):
  for k,m in [(True,2),(31,2),(10241,1),(10240,10240)]:
   with self.assertRaises(ValueError):bulk.contract('matrixdot',k,m,'shared',b'',b'')
 def test_extent_nonfinite(self):
  for raw in (bytes(255),bytes.fromhex('0000807f')+bytes(252)):
   with self.assertRaises(ValueError):bulk.contract('matrixdot',32,2,'shared',raw,bytes(128))
 def test_receipt_mutations(self):
  for key,value in [('K',64),('M',1),('op','fma'),('mode','paired'),('output_floats',3),('rounding','up'),('flush_to_zero',True),('denormals_are_zero',True),('device_intrinsics_qualified',True),('model_math_qualified',True)]:
   row=self.receipt();row[key]=value
   with self.assertRaises(ValueError):bulk.validate_receipt(row,'matrixdot',32,2,'shared',8)
 def obj(self):
  return dict(Name='/owned',Image=q.IMAGE,Config=dict(Image=q.IMAGE,Cmd=q.compile_recipe(Path('/tmp/bulk-build'))[2][-2:],Labels={'b70.hc35bulk.compile':bulk.sha(bulk.SOURCE_PLAN)},User='1000:1000',Entrypoint=['/bin/bash']),HostConfig=dict(NetworkMode='none',NanoCpus=2000000000,Memory=2<<30,MemorySwap=2<<30,Privileged=False,Devices=[],DeviceRequests=[]),Mounts=[dict(Source=str(q.HERE),Destination='/source',RW=False,Type='bind'),dict(Source='/tmp/bulk-build',Destination='/out',RW=True,Type='bind')])
 def test_observed_positive(self):q.observed_compile(self.obj(),'/tmp/bulk-build','owned')
 def test_observed_resources(self):
  for key,value in [('NanoCpus',0),('Memory',0),('MemorySwap',0),('NetworkMode','host'),('Privileged',True),('Devices',[{}]),('DeviceRequests',[{}])]:
   row=self.obj();row['HostConfig'][key]=value
   with self.assertRaises(ValueError):q.observed_compile(row,'/tmp/bulk-build','owned')
 def test_mount_image(self):
  for change in ('mount','image','user'):
   row=self.obj()
   if change=='mount':row['Mounts'].append(dict(Source='/models',Destination='/models',RW=False,Type='bind'))
   elif change=='image':row['Config']['Image']='foreign'
   else:row['Config']['User']='root'
   with self.assertRaises(ValueError):q.observed_compile(row,'/tmp/bulk-build','owned')
 def test_command(self):
  name,argv,cmd=q.compile_recipe(Path('/tmp/bulk-build'),123);self.assertIn('--cpus',cmd);self.assertEqual(cmd[cmd.index('--cpus')+1],'2');self.assertTrue(all(f in argv for f in ('-fno-fast-math','-ffp-contract=off','-frounding-math')));self.assertNotIn('--device',cmd)
 def test_source(self):bulk.source_binding()
if __name__=='__main__':unittest.main()
