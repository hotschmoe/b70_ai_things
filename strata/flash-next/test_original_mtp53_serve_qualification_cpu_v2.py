import subprocess,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import run_original_mtp53_serve_qualification_v2 as q
class Interactive(unittest.TestCase):
 def test_actual_Docker_CLI_and_constructed_stdin(self):
  help=subprocess.run(['docker','run','--help'],capture_output=True,text=True,check=True).stdout
  self.assertIn('--interactive',help);self.assertIn('Keep STDIN open',help)
  base={'image':q.RUNTIME,'workdir':'/','read_only_mounts':{}}
  r={'native_env':{},'native_argv':['/original-serve/original-mtp53-serve','--serve'],'additional_readonly_mounts':{}}
  with patch.object(q.os,'stat',return_value=SimpleNamespace(st_gid=44)):
   argv=q.native_command(base,r,Path('/sdk'),Path('/results'),0)[0]
  self.assertEqual(argv[:3],['docker','run','-i']);self.assertNotIn('-d',argv);self.assertNotIn('--detach',argv)
 def test_input_parser_actual_GGUF_ids_spelling(self):
  self.assertEqual(q.request_command(1,0,[248045,8678,198],1),'BGEN 0 64 fresh=1 pin=2 ckpt=1 temperature=0 rid=1 248045 8678 198')
