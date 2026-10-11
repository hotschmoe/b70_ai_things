"""Owned tiny vectors only; no model/reference execution."""
import ast,math,unittest
from pathlib import Path
import numpy as np
import source44_num38_core_comparator_v1 as c

class Controls(unittest.TestCase):
    def test_owned_targets_and_no_tolerance(self):
        core=np.full((48,128),0.25,dtype=np.float64);owned=c.fp64_core_targets(core,1e-6)
        self.assertTrue(np.all(owned['gdn_rms_square_sum']==8.0))
        result=c.comparisons(owned,{k:np.asarray(v,dtype='<f4')for k,v in owned.items()})
        self.assertTrue(all(v['bitwise_equal']for v in result['comparisons'].values()))
        self.assertFalse(result['captured_inputs_used']);self.assertFalse(result['existing_F32_full48_reference_replaced'])
        bad={k:np.asarray(v,dtype='<f4').copy()for k,v in owned.items()};bad['gdn_core_scaled'][0,0]=0.5
        self.assertFalse(c.comparisons(owned,bad)['comparisons']['gdn_core_scaled']['bitwise_equal'])

    def test_conditional_source_order_tiny_geometry(self):
        q=np.array([1,-1,2,.5],dtype=np.float32);k=np.array([.5,1,0,-1],dtype=np.float32)
        v=np.array([[1]*4,[2]*4,[3]*4,[4]*4],dtype=np.float32)
        h=np.concatenate([q,k,v.reshape(-1)]);result=c._conditional_core(h,np.full(4,.25,dtype=np.float32),.5,4,1,4)
        self.assertTrue(np.array_equal(result,np.broadcast_to(np.array([-.125,-.25,-.375,-.5],dtype=np.float32)[:,None],(4,4))))
        with self.assertRaises(ValueError):c._conditional_core(h,np.array([1,1,1,float('nan')],dtype=np.float32),.5,4,1,4)

    def test_negative_zero_before_is_not_own_zero(self):
        before=bytearray(3145728);before[3]=128
        with self.assertRaises(ValueError):c.conditional_core(None,None,before,1,None)

    def test_primary_original_source_remains_unchanged(self):
        source=Path(c.original.__file__).read_bytes();node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)and n.name=='independent_first')
        self.assertEqual([a.arg for a in node.args.args],['model']);self.assertIn("model.mixer(hc['mixed']",ast.unparse(node))
        first=Path(c.__file__).parent/'original_first_gdn_layer_v1.py'
        self.assertIn("'core':core",first.read_text())

if __name__=='__main__':unittest.main()
