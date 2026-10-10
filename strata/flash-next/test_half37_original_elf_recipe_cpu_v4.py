"""Original code extraction publication/recollection with source-shaped mocks."""
import tempfile,unittest,json
from pathlib import Path
from unittest.mock import patch
import half37_original_elf_spirv_v4 as p
import half37_original_elf_text_recipe_v4 as recipe
class Controls(unittest.TestCase):
 def test_original_extraction_exact_roster_and_substitution_refusal(self):
  raw=p.consume(p.ROOT/'build/owned-indexer-half37',p.MAX_ELF);binding={'CPU':'source-shaped-original-proof'}
  with tempfile.TemporaryDirectory()as tmp,patch.object(p,'original_binding',return_value=(binding,raw)):
   out=Path(tmp)/'extracted';p.extract(out);self.assertTrue(p.finalized_binding(out)['original_embedded_images_extracted']);f=out/'module-33.spv';f.write_bytes((out/'module-36.spv').read_bytes());self.assertRaises(ValueError,p.finalized_binding,out)
 def test_extra_symlink_or_duplicate_metadata_refused(self):
  raw=p.consume(p.ROOT/'build/owned-indexer-half37',p.MAX_ELF);binding={'CPU':'source-shaped-original-proof'}
  with tempfile.TemporaryDirectory()as tmp,patch.object(p,'original_binding',return_value=(binding,raw)):
   out=Path(tmp)/'extracted';p.extract(out);(out/'foreign').write_bytes(b'');self.assertRaises(ValueError,p.finalized_binding,out);(out/'foreign').unlink();f=out/'module-33.spv';saved=f.read_bytes();f.unlink();original=Path(tmp)/'original';original.write_bytes(saved);f.symlink_to(original);self.assertRaises(ValueError,p.finalized_binding,out)
 def test_schema_scope_and_changed_consumed_report_refused(self):
  raw=p.consume(p.ROOT/'build/owned-indexer-half37',p.MAX_ELF);binding={'CPU':'source-shaped-original-proof'}
  with tempfile.TemporaryDirectory()as tmp,patch.object(p,'original_binding',return_value=(binding,raw)):
   out=Path(tmp)/'extracted';p.extract(out);f=out/'extraction.json';saved=f.read_bytes()
   for key,value in [('schema',True),('conversion_removed_before_embedded_SPIRV_claimed',True),('original_runtime_JIT_ISA_observed',True)]:
    record=json.loads(saved);record[key]=value;f.write_text(json.dumps(record));self.assertRaises(ValueError,p.finalized_binding,out)
   f.write_bytes(saved);actual_consume=p.consume;calls=0
   def changing(path,limit):
    nonlocal calls
    result=actual_consume(path,limit)
    if Path(path)==f:
     calls+=1
     if calls==2:return result+b' '
    return result
   with patch.object(p,'consume',side_effect=changing):self.assertRaises(ValueError,p.finalized_binding,out)
 def test_supported_original_text_recipe_and_distinct_unqualified_IGC_probe(self):
  with patch.object(p,'finalized_binding',return_value={'original_helper_sha256':p.HELPER_SHA}),patch.object(recipe.inventory,'inventory_binding',return_value={}):
   result=recipe.text_recipes('/tmp/original-images','/tmp/original-text',123)
  self.assertEqual([r['index']for r in result['commands']],list(p.MODULES));self.assertTrue(result['input_is_original_executed_embedded_SPIRV']);self.assertFalse(result['actual_JIT_ISA_observed'])
  for row in result['commands']:
   self.assertEqual(row['argv'][1],'--to-text');self.assertIn('/images/module-'+str(row['index'])+'.spv',row['argv']);self.assertNotIn('-spirv-max-version=1.5',row['argv'])
  command=recipe.offline_IGC_inventory_recipe('/tmp/inventory',123);self.assertIn(recipe.q.RUNTIME_IMAGE,command);self.assertIn('SYCL_CACHE_PERSISTENT=0',command);self.assertIn('--help',command[-1]);self.assertNotIn(' -device ',command[-1])
if __name__=='__main__':unittest.main()
