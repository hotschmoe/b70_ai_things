#!/usr/bin/env python3
"""Real local HTTP/SSE client controls with synthetic responses, no model/GPU."""
import json,tempfile,threading,time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from batch_api_client_v2 import cohort

class Handler(BaseHTTPRequestHandler):
 protocol_version='HTTP/1.1'
 def log_message(self,*args):pass
 def do_GET(self):
  raw=json.dumps({'data':[{'id':'hotschmoe-dd'},{'id':'fixture-only'}]}).encode();self.send_response(200);self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
 def do_POST(self):
  row=json.loads(self.rfile.read(int(self.headers['Content-Length'])));self.server.requests.append(row)
  self.send_response(200);self.send_header('Content-Type','text/event-stream');self.send_header('Connection','close');self.end_headers();self.close_connection=True
  try:
   for i in range(6):self.wfile.write(('data: '+json.dumps({'choices':[{'delta':{'content':'fixture'}}]})+'\n\n').encode());self.wfile.flush();time.sleep(.02)
   self.wfile.write(b'data: [DONE]\n\n');self.wfile.flush()
  except (BrokenPipeError,ConnectionResetError):pass

def main():
 server=ThreadingHTTPServer(('127.0.0.1',0),Handler);server.requests=[];thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start();url='http://127.0.0.1:'+str(server.server_port)
 try:
  with tempfile.TemporaryDirectory() as name:
   root=Path(name);gate=root/'gate';gate.write_text('synthetic CPU event only\n')
   for n in (2,4,6):
    messages=[[{'role':'user','content':'CPU-only fixture '+str(i)}] for i in range(n)];result=cohort(url,'hotschmoe-dd',messages,root/('cohort'+str(n)),cancel_index=0,max_new=[64]+[8]*(n-1),timeout=5,cancel_gate=gate)
    assert result['client_transport_completed'] and result['real_client_cancel_requested'] and result['rows'][0]['cancel_requested'] and not result['rows'][0]['done_received']
    assert all(row['done_received'] and not row['cancel_requested'] for row in result['rows'][1:])
    assert result['native_cancellation_or_migration_qualified'] is False and result['actual_native_multrow_overlap_qualified'] is False
   assert len({r['user'] for r in server.requests})==12
 finally:server.shutdown();server.server_close();thread.join(timeout=2)
 print(json.dumps({'passed':True,'actual_HTTP_SSE_cpu_cohorts':[2,4,6],'actual_socket_reader_closes':3,'GPU_or_native_engine_requests':0,'scope':'Synthetic local CPU server only; native/API serving mathematics, concurrency and lifecycle unqualified'},ensure_ascii=True))
if __name__=='__main__':main()
