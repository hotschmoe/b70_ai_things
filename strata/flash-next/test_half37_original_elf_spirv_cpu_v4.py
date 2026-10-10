"""Bounded original code and synthetic mutation tests; no runtime/arrays."""
import struct,unittest
from pathlib import Path
import half37_original_elf_spirv_v4 as p
class Controls(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.raw=p.consume(p.ROOT/'build/owned-indexer-half37',p.MAX_ELF)
 def test_actual_original_helperSHA_six_entrypoints_and_instruction_extents(self):
  self.assertEqual(p.digest(self.raw),p.HELPER_SHA);images=p.elf_images(self.raw);self.assertEqual(set(images),set(p.MODULES));self.assertEqual([sum(row['opcode']==115 for row in images[i]['SPIRV']['instructions'])for i in p.MODULES],[2,1,1,2,1,1])
 def test_changed_entrypoint_image_refused(self):
  raw=self.raw.replace(b'_ZTS16Half37Expression\0',b'_ZTS16Half37ExpreXsion\0');self.assertNotEqual(raw,self.raw);self.assertRaises(ValueError,p.elf_images,raw)
 def test_foreign_magic_or_truncated_ELF_refused(self):
  for raw in (self.raw[:63],b'BAD!'+self.raw[4:],self.raw[:-100]):self.assertRaises(ValueError,p.elf_images,raw)
 def test_symbol_offset_substitution_outside_file_or_section_refused(self):
  raw=bytearray(self.raw);offset=struct.unpack_from('<Q',raw,40)[0];size,count,names=struct.unpack_from('<HHH',raw,58);headers=[struct.unpack_from('<IIQQQQIIQQ',raw,offset+i*size)for i in range(count)]
  for h in headers:
   if h[1]!=2:continue
   st=headers[h[6]];strings=raw[st[4]:st[4]+st[5]]
   for o in range(h[4],h[4]+h[5],h[9]):
    n=struct.unpack_from('<I',raw,o)[0];end=strings.find(b'\0',n)
    if strings[n:end]==b'.sycl_offloading.33.data':struct.pack_into('<Q',raw,o+8,2**63);self.assertRaises(ValueError,p.elf_images,bytes(raw));return
  self.fail('Actual original symbol not found')
 def test_zero_or_overrun_SPIRV_instruction_refused(self):
  image=p.elf_images(self.raw)[33]['raw']
  for word in (0,65535<<16|15):
   bad=bytearray(image);struct.pack_into('<I',bad,20,word);self.assertRaises(ValueError,p.spirv,bytes(bad))
 def test_duplicate_entrypoint_refused(self):
  raw=p.elf_images(self.raw)[33]['raw'];words=struct.unpack('<'+'I'*(len(raw)//4),raw);i=5
  while i<len(words):
   n,op=words[i]>>16,words[i]&65535
   if op==15:self.assertRaises(ValueError,p.spirv,raw+struct.pack('<'+'I'*n,*words[i:i+n]));return
   i+=n
  self.fail('Original entrypoint missing')
if __name__=='__main__':unittest.main()
