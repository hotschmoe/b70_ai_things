import struct,unittest
import full_cache_shared_persisted_v2 as p

class Persisted(unittest.TestCase):
 def header(self):
  h=bytearray(b'STRSESS\x01'+struct.pack('<IIQQQQQQ',1,64,17,19,100,0,0,0));struct.pack_into('<Q',h,56,p.header_hash56(bytes(h[:56])));return bytes(h)
 def test_only_fingerprint_and_checksum_are_changed(self):
  original=self.header();changed,b=p.wrong_model_header(original,180)
  self.assertEqual(struct.unpack_from('<Q',changed,16)[0],16);self.assertEqual(changed[:16],original[:16]);self.assertEqual(changed[24:56],original[24:56]);self.assertEqual(struct.unpack_from('<Q',changed,56)[0],p.header_hash56(changed[:56]));self.assertFalse(b['native_checksum_implementation_qualified']);self.assertFalse(b['actual_refusal_observed'])
 def test_invalid_original_header_and_extent_fail_closed(self):
  for offset in [0,8,12,40,56]:
   changed=bytearray(self.header());changed[offset]^=1
   with self.subTest(offset=offset),self.assertRaises(ValueError):p.wrong_model_header(bytes(changed),180)
  for size in [179,True]:
   with self.assertRaises(ValueError):p.wrong_model_header(self.header(),size)
 def test_real_source_400_and_parallel_501_branches_are_distinct(self):
  body={'error':{'code':400,'type':'invalid_request_error','kind':'invalid','message':'session file: saved with another model (model fingerprint differs)'}}
  self.assertFalse(p.fingerprint_refusal(400,body,False)['full_cache_runtime_qualified'])
  for message in ['session file: header checksum mismatch','slot save/restore is disabled (start the server with --slot-save-path DIR)']:
   changed={'error':{**body['error'],'message':message}}
   with self.assertRaises(ValueError):p.fingerprint_refusal(400,changed,False)
  parallel={'error':{'code':501,'type':'server_error','message':'slot save/restore is not available with parallel requests ("parallel" / --batch)'}}
  self.assertTrue(p.fingerprint_refusal(501,parallel,True)['actual_HTTP_refusal_shape_matched'])
  with self.assertRaises(ValueError):p.fingerprint_refusal(501,parallel,False)

if __name__=='__main__':unittest.main()
