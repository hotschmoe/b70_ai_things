"""Actual-shaped native namespace/rawstat and independent decode negatives."""
import copy,tempfile,unittest,json
from pathlib import Path
from source43_api_live_owner_decode_v1 import owner_binding,decoded_output_binding,decode_requests,recipe
import test_source43_api_runtime_cpu_v1 as runtime_controls
class Controls(unittest.TestCase):
 def owner(self):
    state={'Running':True,'Pid':400};obj={'Id':'container','Name':'/owned','HostConfig':{'PidMode':''},'Config':{'Image':'image','Labels':{'b70.c1.prepared':'digest'}},'State':state,'Mounts':[{'Destination':'/build','Source':'/sdk/build','RW':False}]};raw='401 (strata) S '+' '.join(['0']*18+['123','0']);return {'schema':1,'epoch':100,'before_inspection':obj,'after_inspection':copy.deepcopy(obj),'container_host_pid':400,'container_start_ticks':100,'container_status':'NSpid:\t400\t1\n','container_namespace':'pid:[1234]','children_raw':'401','children':[{'host_pid':401,'start_ticks':123,'start_ticks_after':123,'stat_before':raw,'stat_after':raw,'status':'NSpid:\t401\t2\n','NSpid':[401,2],'namespace':'pid:[1234]','exe':'/sdk/build/strata'}],'trace_native_pid':2},{'executable':'/sdk/build/strata','runtime':{'image':'image'}},{'container':'owned','prepared_sha256':'digest'}
 def test_real_shaped_namespace_owner(self):
    p,m,l=self.owner();self.assertEqual(owner_binding(p,m,l)['host_pid'],401)
 def test_foreign_namespace_PID_exe_owner(self):
    for mutate in(lambda p:p.update(trace_native_pid=3),lambda p:p['children'][0].update(namespace='foreign'),lambda p:p['children'][0].update(exe='/foreign'),lambda p:p['children'][0].update(start_ticks_after=124),lambda p:p['children'][0].update(status='NSpid: 999 2\n'),lambda p:p['after_inspection'].update(Id='foreign'),lambda p:p.update(container_host_pid=True)):
     p,m,l=self.owner();mutate(p);self.assertRaises(ValueError,owner_binding,p,m,l)
 def decode(self):
    req=[{'case':i,'repeat':i+1,'tokens':[1,248046],'content':'text'+str(i)}for i in range(8)];hashes={n:'0'*64 for n in('vocab.json','merges.txt','token_type.json','tokenizer.json','chat_template.jinja')};obs={'actual_GPU_touch':False,'actual_model_payload_read':False,'fixtures':[],'tokenizer_file_sha256':hashes,'decoded_outputs':[{'case':r['case'],'repeat':r['repeat'],'text':r['content'],'matches':True}for r in req]};return req,obs,{'tokenizer_files':hashes}
 def test_all_eight_independent_native_text(self):
    req,obs,m=self.decode();self.assertTrue(decoded_output_binding(req,obs,m))
 def test_wrong_text_source_counts_and_order(self):
    for change in(lambda o:o['decoded_outputs'][0].update(text='foreign'),lambda o:o['decoded_outputs'][0].update(matches=1),lambda o:o['decoded_outputs'].pop(),lambda o:o['decoded_outputs'].reverse(),lambda o:o.update(actual_model_payload_read=True),lambda o:o['tokenizer_file_sha256'].update(**{'vocab.json':'f'*64})):
     req,obs,m=self.decode();m=copy.deepcopy(m);change(obs);self.assertRaises(ValueError,decoded_output_binding,req,obs,m)
 def test_original_native_arrays_bound_to_HTTP_text(self):
    with tempfile.TemporaryDirectory()as td:
     root=Path(td);runtime_controls.Controls().fixture(root);requests=decode_requests(root);self.assertEqual(len(requests),8);self.assertEqual(requests[0]['tokens'],[200,248046]);p=root/'api-client/report.json';v=json.loads(p.read_text());v['results'][0]['response']['choices'][0]['message']['reasoning_content']='unqualified';p.write_text(json.dumps(v));self.assertRaises(ValueError,decode_requests,root)
 def test_actual_metadata_command_no_model_devices(self):
    m={'source':'/sdk/source','pack':'/pack','runtime':{'image':'image'}};name,cmd=recipe(Path('/tmp/only-metadata'),m,4321,'0'*64);self.assertIn('/results/output-id-requests.json',cmd);self.assertNotIn('--device',cmd);self.assertNotIn('/model',cmd);self.assertEqual(cmd[cmd.index('--network')+1],'none');self.assertEqual(cmd[cmd.index('--memory')+1],'512m')
if __name__=='__main__':unittest.main()
