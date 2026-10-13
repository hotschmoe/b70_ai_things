"""Tiny software construction tests; no archive/weights/compiler/device reads."""
import unittest
from pathlib import Path
import prepare_prefill_gdn_projection_operator_v1 as source
class Controls(unittest.TestCase):
 def test_actual_production_commands(self):
  c,l=source.commands((source.SDK/'build/incremental-parent-commands.txt').read_text());self.assertEqual(c[-4:],['-o','/out/fixture.o','-c','/inputs/fixture.cpp']);self.assertIn('/sdk/build/libstrata_kernels.a',l);self.assertIn('/sdk/build/libstrata_core.a',l);self.assertIn('-cl-fp32-correctly-rounded-divide-sqrt',l)
 def test_actual_post_source_and_patch(self):
  p=source.SDK/'source/sycl/src/prefill/kernels.dp.cpp';x=source.extract(p.read_text());self.assertEqual(x.count('if constexpr(StoreF32)y[at]=v;'),1);self.assertIn('wsum[0] + wsum[1] + wsum[2] + wsum[3]',x);self.assertIn('sycl::rsqrt',x);self.assertIn('PostStoreFixtureV1<StoreF32>',x)
 def test_actual_fixture_scope(self):
  x=(source.ROOT/'kernels/prefill_gdn_decoder_projection_fixture_v1.cpp').read_text();self.assertIn('for(int T:{1,7,8,9})',x);self.assertIn('const int last=(T-1)%8',x);self.assertIn('last*2880',x);self.assertIn('last*6912',x);self.assertIn('owner.copy(a.input_packet,2880)==owner.copy(b.input_packet,2880)',x);self.assertIn('conv_recurrence_qualified=0',x);self.assertIn('std::_Exit(86)',x);self.assertLess(x.index('q.wait_and_throw();}catch(...){retain();}for(auto*g:execs)'),x.index('for(auto*b:buffers)'))
if __name__=='__main__':unittest.main()
