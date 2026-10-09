#!/usr/bin/env python3
"""Source-derived host branch/ordering tests; no compiler/GPU/model payloads."""
import itertools,re,unittest
from pathlib import Path
PATCH=Path(__file__).parent/'patches/0032-sycl-current-ple-staging-before-nohost-mirror-graph.patch'


def source_predicate():
 patch=PATCH.read_text();match=re.search(r'^\+    const bool ple_precollected = ([^;]+);$',patch,re.M);assert match
 return match[1]

def precollect(do_ple,no_host,ar,device_plan=False):
 expression=source_predicate().replace('ar_on()','ar').replace('&&',' and ').replace('||',' or ').replace('!','not ')
 return eval(expression,{'__builtins__':{}},dict(do_ple=do_ple,no_host=no_host,ar=ar,device_plan_=device_plan))


def schedule(do_ple,no_host,ar,rows,token_base=19,failed=False,host_state=None,device_plan=False):
 if not 1<=rows<=8:raise ValueError('Actual native window bound')
 events=[];host=[-99]*rows if host_state is None else list(host_state);current=list(range(token_base,token_base+rows));collect_count=0
 def gather():
  nonlocal host,collect_count
  events.append('gather_current');collect_count+=1
  if failed:return False
  host=current;return True
 if precollect(do_ple,no_host,ar,device_plan):
  if not gather():return dict(events=events,launched=False,gather_count=collect_count)
  events+=['host_fence','sfence']
 events.append('graph_launch')
 if ar:
  # Existing zero-doorbell graph waits for the later PLE flag.
  events.append('GPU_wait_PLE_flag')
  if do_ple:
   if not gather():return dict(events=events,launched=True,gather_count=collect_count)
  events.append('publish_PLE_flag')
 elif not no_host:
  events.append('GPU_wait_layer0_doorbell')
  if do_ple and not precollect(do_ple,no_host,ar,device_plan):
   if not gather():return dict(events=events,launched=True,gather_count=collect_count)
  events.append('publish_layer0_doorbell')
 if do_ple:events.append('GPU_copy_current_PLE')
 return dict(events=events,launched=True,gather_count=collect_count,host_image=host,current_image=current)


class StagingTests(unittest.TestCase):
 def test_actual_added_cpp_predicate_and_store_order(self):
  source=PATCH.read_text();self.assertEqual(source_predicate(),'do_ple && !ar_on() && (device_plan_ || no_host)');self.assertIn('+        if (k == 0 && do_ple && !ple_precollected)',source);self.assertLess(source.index('+        if (!ss.ple.table->gather_batch'),source.index('     if (trace_h_ != nullptr)'));self.assertIn('+        std::atomic_thread_fence',source);self.assertIn('+        _mm_sfence();',source);self.assertIn(' return false;',source)
 def test_branch_matrix_all_supported_windows_and_mirror_labels(self):
  for do_ple,no_host,ar,mirror,rows in itertools.product((False,True),(False,True),(False,True),(False,True),range(1,9)):
   if mirror and ar:continue # all-VRAM ar_on excludes native mirror tier.
   out=schedule(do_ple,no_host,ar,rows,device_plan=mirror);self.assertEqual(out['gather_count'],int(do_ple))
   if do_ple:
    self.assertEqual(out['host_image'],out['current_image']);self.assertLess(out['events'].index('gather_current'),out['events'].index('GPU_copy_current_PLE'))
   if do_ple and (no_host or mirror) and not ar:self.assertLess(out['events'].index('gather_current'),out['events'].index('graph_launch'))
 def test_device_plan_hostloop_also_precollects_and_never_double_gathers(self):
  for rows in range(1,9):
   out=schedule(True,False,False,rows,device_plan=True);self.assertEqual(out['gather_count'],1);self.assertLess(out['events'].index('gather_current'),out['events'].index('graph_launch'))
 def test_failure_new_route_before_graph(self):
  for rows in range(1,9):
   out=schedule(True,True,False,rows,failed=True);self.assertFalse(out['launched']);self.assertNotIn('graph_launch',out['events'])
 def test_warm_cached_graph_gets_each_new_host_image(self):
  first=schedule(True,True,False,4,token_base=19);second=schedule(True,True,False,4,token_base=29,host_state=first['host_image']);self.assertNotEqual(first['host_image'],second['host_image']);self.assertEqual(second['host_image'],second['current_image']);self.assertEqual(second['gather_count'],1)
 def test_pair_stage_only_layer1_owner_collects(self):
  for bounds in ([(0,32),(32,48)],[(0,1),(1,48)]):
   counts=[schedule(lo<=1<hi,True,False,1)['gather_count'] for lo,hi in bounds];self.assertEqual(sum(counts),1)
 def test_actual_window_bound_and_rejection(self):
  header=Path('/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261009T185206Z-d5q32zc7/source/include/strata/kernels/verify_kernels.hpp').read_text();self.assertIn('kVerifyMaxT = 8',header)
  for rows in (0,9,16):
   with self.assertRaises(ValueError):schedule(True,True,False,rows)
 def test_old_missing_branch_negative(self):
  do_ple,no_host,ar=True,True,False;old_collect=int(do_ple and (ar or not no_host));self.assertEqual(old_collect,0);self.assertTrue(precollect(do_ple,no_host,ar))
 def test_legacy_doorbell_and_zero_doorbell_orders_preserved(self):
  for no_host,ar in ((False,False),(False,True),(True,True)):
   out=schedule(True,no_host,ar,2);self.assertLess(out['events'].index('graph_launch'),out['events'].index('gather_current'));self.assertEqual(out['gather_count'],1)

if __name__=='__main__':unittest.main()
