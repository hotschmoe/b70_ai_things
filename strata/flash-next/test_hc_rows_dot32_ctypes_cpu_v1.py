import ctypes,hashlib,os,subprocess,tempfile,unittest
from pathlib import Path
import numpy as np
import original_native_hc_bulk_f32_v1 as a
HERE=Path(__file__).resolve().parent
FLAGS=['-O3','-std=c++17','-shared','-fPIC','-mfma','-fno-fast-math','-ffp-contract=off','-frounding-math']
class Controls(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  supplied=os.environ.get('HCROWS_CPU_ELF');expected=os.environ.get('HCROWS_CPU_ELF_SHA256')
  if supplied:
   if not expected:raise ValueError('Supplied opt-in CPU ELF requires explicit SHA')
   cls.tmp=None;cls.elf=Path(supplied);cls.sha=expected
  else:
   cls.tmp=tempfile.TemporaryDirectory(prefix='hc-rows-CPU-only-');cls.elf=Path(cls.tmp.name)/'hc-rows.so';cls.command=['g++',*FLAGS,str(HERE/'hc_rows_dot32_ctypes_v1.cpp'),'-o',str(cls.elf),'-lm'];subprocess.run(cls.command,check=True,capture_output=True,timeout=60);cls.sha=hashlib.sha256(cls.elf.read_bytes()).hexdigest()
  cls.backend=a.RowsDot32(cls.elf,cls.sha)
 @classmethod
 def tearDownClass(cls):
  if cls.tmp is not None:cls.tmp.cleanup()
 def test_shared_and_paired_exact_scalar(self):
  w=(((np.arange(4*64)*17+11)%31)-15).astype('<f4')/np.float32(16);w=w.reshape(4,64);x=(((np.arange(4*64)*13+7)%23)-11).astype('<f4')/np.float32(8);x=x.reshape(4,64)
  for paired in(False,True):
   operand=x if paired else x[0];expected=np.array([a.scalar.dot32(row,x[i]if paired else operand)for i,row in enumerate(w)],dtype='<f4');self.assertEqual(self.backend.rows(w,operand,paired).tobytes(),expected.tobytes())
 def test_FMA_and_subnormal_rows(self):
  w=np.zeros((3,64),dtype='<f4');x=np.zeros((3,64),dtype='<f4');w[0,0]=-1;x[0,0]=1;w[0,32]=1+2**-23;x[0,32]=1-2**-23;w[1,0]=2**-126;x[1,0]=.5;w[2,0]=2**-149;x[2,0]=2**24
  self.assertEqual(self.backend.rows(w,x,True).tobytes(),np.array([-2**-46,2**-127,2**-125],dtype='<f4').tobytes())
 def test_bulk_norm_exact_scalar_candidate(self):
  r=(((np.arange(4*64)*7)%19)-9).astype('<f4').reshape(4,64)/np.float32(8);norm=np.ones_like(r);expected=a.scalar.normalize_streams(r,norm,1e-6);actual=self.backend.normalize(r,norm,1e-6)
  for name in expected:self.assertEqual(actual[name].tobytes(),expected[name].tobytes())
 def test_nonRNE_refused_without_reset(self):
  lib=ctypes.CDLL('libm.so.6');get=lib.fegetround;get.restype=ctypes.c_int;setround=lib.fesetround;setround.argtypes=[ctypes.c_int];before=get()
  try:
   self.assertEqual(setround(0x800),0)
   with self.assertRaises(ValueError):self.backend.rows(np.ones((1,32),dtype='<f4'),np.ones(32,dtype='<f4'))
   self.assertEqual(get(),0x800)
  finally:self.assertEqual(setround(before),0)
 def test_wrongELF_and_badfinite_refused(self):
  with self.assertRaises(ValueError):a.RowsDot32(self.elf,'0'*64)
  w=np.ones((1,32),dtype='<f4');w[0,0]=np.nan
  with self.assertRaises(ValueError):self.backend.rows(w,np.ones(32,dtype='<f4'))
if __name__=='__main__':unittest.main()
