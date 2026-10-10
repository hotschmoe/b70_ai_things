import importlib.util,json,tempfile,unittest,threading,time,socket,errno,ast
from pathlib import Path
from unittest.mock import patch,Mock
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('cpu_pilot',HERE/'qualify_cpu_functional_pilot_v3.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class PilotTests(unittest.TestCase):
    def response(self,case=1):
        return {'stop':True,'stop_type':'eos','truncated':False,'tokens_evaluated':3,'prompt':'rendered','tokens':[12,99],'tokens_predicted':2,'generation_settings':{'temperature':0,'seed':1234,'n_predict':64,'repeat_penalty':1,'samplers':['temperature'],'ignore_eos':False},'content':'12' if case else 'def add(a, b):\n    return a + b\n'}
    def test_known_functions_no_generated_execution(self):
        self.assertTrue(m.functional(0,'def add(a, b):\n    return a + b'))
        self.assertTrue(m.functional(0,'```python\ndef add(a, b):\n    return b + a\n```'))
        for text in ['def add(a,b):\n return a-b','import os\ndef add(a,b):\n return a+b','def add(a,b):\n print(a)\n return a+b','@evil\ndef add(a,b):\n return a+b']:
            with self.assertRaises(ValueError):m.functional(0,text)
    def test_post_source_preserves_both_pages_even_hash_failure(self):
        page=Mock();page.preserve.return_value={'passed':True}
        with patch.object(m,'model_identity',side_effect=ValueError('full-four mismatch')):
            with self.assertRaises(ValueError):m.post_identity(page,Path('/third'),'/model',{},Path('/out'),10)
        self.assertEqual(page.preserve.call_count,2)
        page.reset_mock();page.preserve.side_effect=[{'passed':False},{'passed':True}]
        with patch.object(m,'model_identity',return_value={'passed':True,'started_epoch':11}) as scan:
            with self.assertRaises(ValueError):m.post_identity(page,Path('/third'),'/model',{},Path('/out'),10)
            scan.assert_called_once()
        self.assertEqual(page.preserve.call_count,2)

    def test_known_pages_exact_third_shard_roster(self):
        paths=[Path('/original/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-%05d-of-00004.gguf'%i) for i in range(1,5)]
        self.assertEqual(m.known_page_target(paths),paths[2])
        bad=paths.copy();bad[1],bad[2]=bad[2],bad[1]
        with self.assertRaises(ValueError):m.known_page_target(bad)
        with self.assertRaises(ValueError):m.known_page_target(paths[:3])
        bad=paths.copy();bad[2]=Path('/original/other.gguf')
        with self.assertRaises(ValueError):m.known_page_target(bad)
    def test_numeric_and_reasoning_failures(self):
        self.assertTrue(m.functional(1,'12\n'))
        for text in ['The answer is 12.','13','<think>12</think>12']:
            with self.assertRaises(ValueError):m.functional(1,text)
    def test_actual_completion_schema_and_bad_finishes(self):
        self.assertTrue(m.response_gate(self.response(),[1,2,3],1,'rendered'))
        for key,val in [('stop_type','limit'),('truncated',True),('tokens_evaluated',4),('prompt','other'),('tokens',[]),('tokens_predicted',64),('generation_settings',{'temperature':0.1,'seed':1234})]:
            r=self.response();r[key]=val
            with self.assertRaises(ValueError):m.response_gate(r,[1,2,3],1,'rendered')
    def test_memory_bound_swap_pressure_oom_negatives(self):
        plan={'minimum_live_available_bytes':6,'memory_cap_bytes':100}
        baseline={'host':{'SwapFree':10}}
        sample={'host':{'MemAvailable':7,'SwapFree':10},'process':{'VmSwap':0},'cgroup':{'memory.max':'100','memory.swap.max':'0','memory.swap.current':'0','memory.events':'max 0\noom 0\noom_kill 0'}}
        self.assertTrue(m.memory_gate(sample,baseline,plan))
        for key,val in [('memory.max','101'),('memory.swap.max','1'),('memory.swap.current','1'),('memory.events','max 0\noom 1\noom_kill 0')]:
            r=json.loads(json.dumps(sample));r['cgroup'][key]=val
            with self.assertRaises(ValueError):m.memory_gate(r,baseline,plan)
        for key,val in [('SwapFree',9),('MemAvailable',5)]:
            r=json.loads(json.dumps(sample));r['host'][key]=val
            with self.assertRaises(ValueError):m.memory_gate(r,baseline,plan)
    def test_only_owned_container(self):
        obj={'Name':'/owned','Image':'image','Config':{'Labels':{'b70.cpu-functional-pilot':'binding'}}}
        self.assertTrue(m.owned(obj,'owned','image','binding'))
        self.assertFalse(m.owned(obj,'different','image','binding'))
        self.assertFalse(m.owned(obj,'owned','other','binding'))
        self.assertFalse(m.owned(obj,'owned','image','different'))
    def test_memory_failure_unblocks_waiting_http_by_owned_stop(self):
        plan={'minimum_live_available_bytes':6,'memory_cap_bytes':100}
        baseline={'host':{'SwapFree':10}}
        failed={'host':{'MemAvailable':5,'SwapFree':10}}
        state={'running':True};request_started=threading.Event();request_unblocked=threading.Event();done=threading.Event()
        samples=[];errors=[];stopping=threading.Event()
        def snapshot(name):
            return {'Name':'/owned','Image':'image','Config':{'Labels':{'b70.cpu-functional-pilot':'binding'}},'State':{'Running':state['running'],'ExitCode':0}}
        def controlled(command,**kwargs):
            self.assertEqual(command,['docker','stop','--time','30','owned'])
            state['running']=False;request_unblocked.set();return Mock(returncode=0)
        def blocked_http():
            request_started.set();request_unblocked.wait(60);done.set()
        request=threading.Thread(target=blocked_http,daemon=True);request.start();self.assertTrue(request_started.wait(1))
        controller=m.OwnedServerStop('owned','image','binding')
        try:
            with patch.object(m,'memory_snapshot',return_value=failed),patch.object(m,'inspect',side_effect=snapshot),patch.object(m.subprocess,'run',side_effect=controlled) as stop:
                started=time.monotonic();monitor=threading.Thread(target=m.monitor_memory,args=(123,stopping,samples,errors,baseline,plan,controller));monitor.start()
                self.assertTrue(done.wait(2),'Blocked HTTP was not released immediately by owned shutdown')
                monitor.join(2);self.assertFalse(monitor.is_alive());self.assertLess(time.monotonic()-started,2)
                self.assertTrue(errors);self.assertTrue(stopping.is_set());self.assertFalse(state['running'])
                controller.stop() # ordinary parent cleanup after monitor-triggered stop
                self.assertEqual(stop.call_count,1)
        finally:request_unblocked.set();request.join(2)
    def test_controlled_stop_refuses_foreign_container(self):
        foreign={'Name':'/owned','Image':'image','Config':{'Labels':{'b70.cpu-functional-pilot':'foreign'}},'State':{'Running':True}}
        with patch.object(m,'inspect',return_value=foreign),patch.object(m.subprocess,'run') as stop:
            with self.assertRaises(ValueError):m.OwnedServerStop('owned','image','binding').stop()
            stop.assert_not_called()
    def test_controlled_stop_idempotent_terminal_cleanup(self):
        state={'running':True}
        def snapshot(name):return {'Name':'/owned','Image':'image','Config':{'Labels':{'b70.cpu-functional-pilot':'binding'}},'State':{'Running':state['running'],'ExitCode':0}}
        def controlled(*args,**kwargs):state['running']=False;return Mock(returncode=0)
        with patch.object(m,'inspect',side_effect=snapshot),patch.object(m.subprocess,'run',side_effect=controlled) as stop:
            controller=m.OwnedServerStop('owned','image','binding');first=controller.stop();second=controller.stop()
            self.assertIs(first,second);self.assertFalse(second['State']['Running']);self.assertEqual(stop.call_count,1)

    def test_exact_two_metadata_fixtures_declared_and_bounded(self):
        plan={'prompts':['first','second']}
        prepared={'fixtures':[{'messages':[{'role':'user','content':x}],'rendered':'render '+x,'ids':[1,2,3]} for x in plan['prompts']]}
        self.assertTrue(m.fixture_gate(plan,prepared))
        for records in [prepared['fixtures'][:1],prepared['fixtures']*2,list(reversed(prepared['fixtures']))]:
            with self.assertRaises(ValueError):m.fixture_gate(plan,{'fixtures':records})
        bad=json.loads(json.dumps(prepared));bad['fixtures'][0]['ids']=[1]*2048
        with self.assertRaises(ValueError):m.fixture_gate(plan,bad)

    def test_production_container_mount_depth_and_all_entry_modes(self):
        source=(HERE/'qualify_cpu_functional_pilot_v3.py').read_text()
        shallow=Path('/harness/pilot.py')
        with self.assertRaises(IndexError):_=shallow.parents[2]
        deep=Path(m.CONTAINER_RUNNER);self.assertEqual(deep.parents[2],Path('/harness'))
        self.assertNotIn("'/harness/pilot.py'",source)
        self.assertNotIn(':/harness/pilot.py:ro',source)
        self.assertIn("CONTAINER_RUNNER,'--inside-tokenizer'",source)
        self.assertIn("CONTAINER_RUNNER,'--inside-server'",source)
        self.assertIn("CONTAINER_RUNNER,'--inside-check'",source)
        self.assertIn('wrapper_version_preflight(plan,a.output,a.build_root,supplement)',source)
        self.assertLess(source.index('wrapper_version_preflight(plan,a.output,a.build_root,supplement)'),source.index("model_identity(plan['model_root'],lock,a.output/'pre-full-four.json')"))
        self.assertLess(source.index('prepared_tokens=metadata_tokenizer(plan,a.output)'),source.index("model_identity(plan['model_root'],lock,a.output/'pre-full-four.json')"))
        cmd=m.server_command('owned',Path('/out'),{'runner_sha256':'binding','memory_cap_bytes':2<<30,'image':'image'},Path('/build'),[],Path('/runner.py'),None)
        self.assertIn('/runner.py:'+m.CONTAINER_RUNNER+':ro',cmd)
        self.assertEqual(cmd[-2:],['--inside-server','/results/launch.json'])
        self.assertIn(m.CONTAINER_RUNNER,cmd)

    def test_real_loopback_time_wait_reused_and_probe_closed(self):
        listener=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        listener.bind(('127.0.0.1',0));port=listener.getsockname()[1];listener.listen(1)
        client=socket.socket(socket.AF_INET,socket.SOCK_STREAM);accepted=None
        try:
            client.connect(('127.0.0.1',port));accepted,_=listener.accept()
            accepted.sendall(b'x');accepted.shutdown(socket.SHUT_WR)
            self.assertEqual(client.recv(1),b'x');self.assertEqual(client.recv(1),b'')
            accepted.close();accepted=None;client.close();listener.close()
            # Observe actual kernel TIME_WAIT state for the server's local port.
            states=[line.split()[3] for line in Path('/proc/net/tcp').read_text().splitlines()[1:] if int(line.split()[1].split(':')[1],16)==port]
            self.assertIn('06',states,'OS control must exercise actual TIME_WAIT')
            with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as bare:
                with self.assertRaises(OSError) as failure:bare.bind(('127.0.0.1',port))
                self.assertEqual(failure.exception.errno,errno.EADDRINUSE)
            m.port_preflight(port)
            # A listener can bind after the preflight, proving its probe closed.
            with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as next_listener:
                next_listener.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
                next_listener.bind(('127.0.0.1',port));next_listener.listen(1)
        finally:
            if accepted:accepted.close()
            client.close();listener.close()
    def test_real_loopback_foreign_listener_rejected(self):
        with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as foreign:
            foreign.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            foreign.bind(('127.0.0.1',0));port=foreign.getsockname()[1];foreign.listen(1)
            with self.assertRaises(OSError) as failure:m.port_preflight(port)
            self.assertEqual(failure.exception.errno,errno.EADDRINUSE)
            # Foreign listener remains open and unaffected by the failed probe.
            with socket.create_connection(('127.0.0.1',port),timeout=1) as client:
                accepted,_=foreign.accept();accepted.close()

    def test_real_httplib_reuseport_only_time_wait(self):
        with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as listener:
            listener.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEPORT,1)
            listener.bind(('127.0.0.1',0));port=listener.getsockname()[1];listener.listen(1)
            with socket.create_connection(('127.0.0.1',port),timeout=1) as client:
                accepted,_=listener.accept()
                accepted.sendall(b'x');accepted.shutdown(socket.SHUT_WR)
                self.assertEqual(client.recv(1),b'x');self.assertEqual(client.recv(1),b'');accepted.close()
        states=[line.split()[3] for line in Path('/proc/net/tcp').read_text().splitlines()[1:] if int(line.split()[1].split(':')[1],16)==port]
        self.assertIn('06',states)
        # This is the failed ADDR-only proposal under the real httplib policy.
        with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as addr_only:
            addr_only.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            with self.assertRaises(OSError):addr_only.bind(('127.0.0.1',port))
        m.port_preflight(port)
    def test_real_foreign_reuseport_listener_rejected(self):
        with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as foreign:
            foreign.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEPORT,1)
            foreign.bind(('127.0.0.1',0));port=foreign.getsockname()[1];foreign.listen(1)
            # A naive reuse-port probe can bind/share the live foreign port.
            with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as unsafe:
                unsafe.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
                unsafe.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEPORT,1)
                unsafe.bind(('127.0.0.1',port))
            with self.assertRaises(OSError) as failure:m.port_preflight(port)
            self.assertEqual(failure.exception.errno,errno.EADDRINUSE)

    def test_all_nonport_functions_and_model_settings_unchanged(self):
        old=ast.parse((HERE/'qualify_cpu_functional_pilot_v2.py').read_text())
        new=ast.parse((HERE/'qualify_cpu_functional_pilot_v3.py').read_text())
        definitions=lambda tree:{n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        before,after=definitions(old),definitions(new)
        self.assertEqual(set(after)-set(before),{'port_preflight'})
        for name in before:
            if name=='execute_case':continue
            self.assertEqual(ast.dump(before[name]),ast.dump(after[name]),name)
        old_plan=json.loads((HERE/'cpu-functional-pilot-source-plan-v2.json').read_text())
        new_plan=json.loads((HERE/'cpu-functional-pilot-source-plan-v3.json').read_text())
        for key in ['server_argv','prompts','image','tokenizer_image','tokenizer_source','tokenizer_pack','model_root','memory_cap_bytes','minimum_start_available_bytes','minimum_live_available_bytes','case_deadline_seconds','research_alias','build_plan_sha256']:
            self.assertEqual(old_plan[key],new_plan[key],key)

    def test_launch_grants_and_cleanenv(self):
        command=m.server_command('owned',Path('/results'),{'runner_sha256':'binding','memory_cap_bytes':100,'image':'image'},Path('/build'),[Path('/model/'+str(i)) for i in range(4)],Path('/pilot.py'),None)
        self.assertNotIn('--device',command);self.assertNotIn('--privileged',command);self.assertNotIn('--group-add',command)
        self.assertEqual(command[command.index('--entrypoint')+1],'/usr/bin/env')
        self.assertIn('-i',command);self.assertEqual(command[command.index('--memory')+1],command[command.index('--memory-swap')+1])
        self.assertEqual(sum(x.endswith(':ro') and x.startswith('/model/') for x in command),4)
    def test_dependency_unresolved_and_gpu_rejected(self):
        for output in ['libgomp.so.1 => not found','libsycl.so => /tmp/libsycl.so (0x123)']:
            with patch.object(m.subprocess,'check_output',return_value=output):
                with self.assertRaises(ValueError):m.dependency_binding('/fake')
    def test_unrelated_failed_build_rejected_before_lifecycle(self):
        build={'return_code':0,'container_removed':True,'container_terminal':{'Running':False,'ExitCode':0,'OOMKilled':False},'actual_GPU_touch':False,'actual_model_payload_read':False,'passed':False,'errors':['unknown compiler failure'],'ELFs':{}}
        with self.assertRaises(ValueError):m.original_build_gate(build,{},'/fake')
    def test_actual_root_build_metadata_only_and_failed_scope(self):
        root=Path('/mnt/vm_8tb/b70/build/flashnext-cpu-build-v3-20261010')
        build=m.read(root/'receipt.json');supp=m.read(root/'elf-closure-v1.json')
        self.assertFalse(build['passed']);self.assertTrue(supp['passed'])
        self.assertTrue(m.original_build_gate(build,supp,root))
        bad=json.loads(json.dumps(supp));bad['ELFs'].pop('llama-server')
        with self.assertRaises(ValueError):m.original_build_gate(build,bad,root)
        bad=json.loads(json.dumps(supp));bad['original_build_receipt_passed']=True
        with self.assertRaises(ValueError):m.original_build_gate(build,bad,root)

if __name__=='__main__':unittest.main()
