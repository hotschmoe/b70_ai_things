import unittest
from pathlib import Path
import prepare_original_mtp53_greedy64_compile_v2 as m
SDK=Path('/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261012T012115Z-_kwur0ww/source/sycl/src/program/generate.cpp')
class OwnedShutdown(unittest.TestCase):
 def test_exact_actual_shutdown_and_draft_retirement(self):
  raw=SDK.read_text();generated=m.transform(raw);hook=generated[generated.index('        if(original53_mode){'):];hook=hook[:hook.index('        // --pipeline-windows 2:')]
  self.assertNotIn('return 0;',hook)
  order=['queues_wait_and_throw();','original53.reset();','token_graph_free(tgraph)','session_graphs_free(gr)','ver.release_stage_mirror_graphs()','vh.stage_mirror.reset(); first_stage_mirror.reset();','trace_serving_graphs_retired()','gguf_src.close();','owner->release();','std::_Exit(0);']
  positions=[hook.index(s)for s in order];self.assertEqual(positions,sorted(positions))
  anchor=raw.index('        save_profile("exit");');start=raw.index('        try {',anchor);end=raw.index('        std::_Exit(0);',start)+len('        std::_Exit(0);');expected=raw[start:end]
  start=hook.index('            try {');end=hook.index('            std::_Exit(0);',start)+len('            std::_Exit(0);');actual='\n'.join(line[4:]for line in hook[start:end].splitlines());actual=actual.replace('            original53.reset(); // drain-protected draft owner retires before any target/context teardown\n','')
  self.assertEqual(actual,expected)
 def test_variables_are_available_and_initialized(self):
  raw=SDK.read_text();hook=raw.index('        // --pipeline-windows 2: stage 0');
  for declaration in ('SessionGraphs gr;','TokenGraph tgraph;','ExpertPool pool(','const bool stage_mirrors=','VerifyHits vh;','Verifier ver_b;'):
   self.assertLess(raw.index(declaration),hook)
  s=m.transform(raw);self.assertIn('if(!stage_mirrors)',s)
  self.assertIn('strata shutdown: mirror release failed: %s\\n',s)
