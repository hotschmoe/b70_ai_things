import unittest
from pathlib import Path
from unittest.mock import patch
import probe_unitrace_indexer40_v2 as p

class PathTests(unittest.TestCase):
 def test_only_directory_output(self):
  c=p.tool_argv();prefix=c[:-len(p.helper_argv())];self.assertNotIn('--output',prefix);self.assertNotIn('--result-dir',prefix)
  self.assertEqual(c[c.index('--output-dir-path')+1],'/results/unitrace');self.assertEqual(c[-len(p.helper_argv()):],p.helper_argv())
 def test_exact_smoke_application(self):
  c=p.tool_argv(['/usr/bin/true']);self.assertEqual(c[-1],'/usr/bin/true');self.assertNotIn('/leaf/indexer-materialized40',c)
 def test_no_device_smoke_recipe(self):
  with patch.object(Path,'stat')as st:
   st.return_value.st_gid=991;c=p.filesystem_smoke_recipe(Path('/tmp/new'),'smoke','a'*64)
  self.assertNotIn('--device',c);self.assertNotIn('--group-add',c);self.assertIn('--inside-filesystem-smoke',c[-1])
  self.assertIn(str(p.INSTALL)+':/tool:ro',c);self.assertIn('/tmp/new:/results:rw',c);self.assertIn('none',c)
 def test_scope_and_unchanged_body(self):
  self.assertEqual(p.MODES,('single_raw','single_materialized','steps_raw','steps_materialized','batch_raw','batch_materialized'))
  self.assertEqual(len(p.FIELDS),8)
 def test_source_admission(self):self.assertEqual(len(p.source_binding()),64)
if __name__=='__main__':unittest.main()
