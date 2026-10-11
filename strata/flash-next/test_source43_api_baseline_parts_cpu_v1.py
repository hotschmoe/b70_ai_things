"""Complete config/preservation and actual two-thread loopback HTTP control."""
import copy,json,tempfile,threading,time,unittest
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import source43_api_baseline_profile_v1 as p
import source43_api_baseline_client_v1 as c
import registry_source43_api174_association_v1 as r
from test_source43_normal_model_cpu_v1 import Controls as Shapes
class Controls(unittest.TestCase):
 def test_completecfg_changes_only_mode_alias_port_manifest(self):
  m,cfg,inputs=Shapes().fixture();got=p.expected_config(cfg,18083);want=copy.deepcopy(cfg);want.update(model_name='hotschmoe-dd',aliases=[p.ALIAS],port=18083,artifact_identity_manifest='/results/artifact-identity.json');want['env']['STRATA_QSA_INDEXER_MATERIALIZED_HALF']='1';self.assertEqual(got,want);self.assertEqual(got['model_name'],'hotschmoe-dd')
 def test_nonmatchedgeometry_refused(self):
  _,cfg,_=Shapes().fixture();cfg['args'][cfg['args'].index('--prefill')+1]='128'
  with self.assertRaises(ValueError):p.expected_config(cfg,18083)
 def test_named174preserves173_and_doesnotfakeoldgate(self):
  with tempfile.TemporaryDirectory()as t:
   path=Path(t)/'registry.yaml';path.write_bytes(r.proposed_bytes());proof=r.association(path);self.assertEqual(proof['actual_models'],174);self.assertFalse(proof['old173_current_global_hash_gate_passed']);path.write_bytes(path.read_bytes().replace(b'weights: Unsloth-UD-Q4_K_XL-original-GGUF',b'weights: FOREIGN',1))
   with self.assertRaises(ValueError):r.association(path)
 def test_genuine_two_HTTP_connections_overlap_and_close(self):
  calls=[];barrier=threading.Barrier(2);lock=threading.Lock()
  class Handler(BaseHTTPRequestHandler):
   def log_message(self,*a):pass
   def do_GET(self):self.reply({'data':[{'id':c.PRIMARY},{'id':p.ALIAS}]})
   def reply(self,data):
    raw=json.dumps(data).encode();self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
   def do_POST(self):
    packet=json.loads(self.rfile.read(int(self.headers['Content-Length'])));text=packet['messages'][-1]['content'];expected='B70_READY'if'B70_READY'in text else'42'if'19 plus 23'in text else'Paris'if'France'in text else'NATIVE_31415'
    with lock:calls.append(threading.get_ident());index=len(calls)
    if index>6:barrier.wait(timeout=5);time.sleep(.03)
    self.reply({'model':packet['model'],'choices':[{'index':0,'message':{'content':expected},'finish_reason':'stop'}],'usage':{'prompt_tokens':10,'completion_tokens':1,'total_tokens':11}})
  server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever);thread.start()
  try:
   with tempfile.TemporaryDirectory()as t:result=c.execute('http://127.0.0.1:'+str(server.server_port),Path(t)/'out');self.assertEqual(len(result['results']),8);self.assertTrue(result['actual_HTTP_overlap_passed']);self.assertFalse(result['engine_token_provenance_qualified']);self.assertFalse(result['owned_teardown_qualified']);self.assertNotEqual(calls[-1],calls[-2])
  finally:server.shutdown();thread.join();server.server_close()
if __name__=='__main__':unittest.main()
