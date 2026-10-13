import struct,tempfile,unittest
from pathlib import Path
from original_mtp53_http_compound_v1 import prefix_metadata
class SerializedOwner(unittest.TestCase):
 def test_real_writer_header_owner_and_prefix(self):
  def n(x):return struct.pack('<Q',x)
  magic=b'ORIGINAL-SHIFTED-CP1';inner=b'ORIGINAL34-CP2-LE4096'
  raw=n(len(magic))+magic+n(2)+struct.pack('<2I',248045,8678)+n(198)+n(1024)+n(len(inner))+inner+n(101)+n(3)+n(2)+n(1)
  with tempfile.TemporaryDirectory()as d:
   p=Path(d)/'original.bin';p.write_bytes(raw);r=prefix_metadata(p);self.assertEqual((r['source_rid'],r['source_slot_generation'],r['target_input_tokens'],r['original_lookahead']),(101,3,2,198))
   p.write_bytes(raw[:-8]+n(0))
   with self.assertRaises(ValueError):prefix_metadata(p)
