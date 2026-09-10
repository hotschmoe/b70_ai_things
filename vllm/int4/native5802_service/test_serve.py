"""CPU-only service gates; all credentials/results/processes are test fixtures."""
import copy
import ast
import hmac
from types import SimpleNamespace
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import serve


class Gates(unittest.TestCase):
    def test_unfinalized_and_old_trial_refused(self):
        with self.assertRaisesRegex(RuntimeError,'not finalized'):
            serve.verify_inputs({'schema':1,'scope':serve.SCOPE,'inputs_finalized':False})
        with self.assertRaisesRegex(RuntimeError,'not qualified'):
            serve.validate_qualification({}, {'qualified':True,'user_authorized_day_trial':True})

    def test_missing_backend200k_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'plan.json';p.write_text(json.dumps({'out':d,'jobs':[]}))
            with self.assertRaisesRegex(RuntimeError,'workloads incomplete'):serve.evidence(p)

    def plan(self,root):
        cfg=['--max-model-len','200000','--max-num-seqs','4','--max-num-batched-tokens','32768','--dtype','float16','--quantization','gptq','--enable-prefix-caching','--compilation-config','{"cudagraph_mode":"FULL_DECODE_ONLY"}']
        (root/'Config.json').write_text(json.dumps({'Cmd':cfg,'Env':[]}))
        (root/'Mounts.json').write_text(json.dumps([{'Source':str(serve.REPO/'models/files/qwen3.8-27b/int4-autoround-gptq-relabel-r212'),'Destination':'/model','RW':False}]))
        return {'server':['backend','--image',serve.IMAGE,'--served-model','hotschmoe-dd','--served-alias','research','--tensor-parallel-size','2','--p2p','0','--mtp','3','--kv-dtype','fp8_e4m3','--hook','load','--scales',str(root/'scale'),'--preservation',str(root),'--name','qualified-arm','--out',str(root/'run')]}

    def test_feature_rejections_and_command_adaptation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);plan=self.plan(root)
            original_read=serve.read
            with patch.object(serve,'sha',return_value=serve.SCALE),patch.object(serve.Path,'read_text',return_value='research'),patch.object(serve,'read',side_effect=lambda p:json.loads(Path(p).read_bytes())):
                serve.verify_plan(plan,200000)
                for flag,bad in [('--image','old7b'),('--p2p','1'),('--mtp','0'),('--served-model','research')]:
                    wrong=copy.deepcopy(plan);wrong['server'][wrong['server'].index(flag)+1]=bad
                    with self.assertRaises(RuntimeError):serve.verify_plan(wrong,200000)
                with patch.dict(os.environ,{'B70_XPU_GDN_PREFIX_CONV_COPY':'1'}):
                    with self.assertRaisesRegex(RuntimeError,'adapter inherited'):serve.verify_plan(plan,200000)
            cmd=serve.service_command(plan,Path('/fresh/server'))
            self.assertEqual(serve.value(cmd,'--port'),'18124');self.assertEqual(serve.value(cmd,'--served-model'),'hotschmoe-dd')
            restored=cmd[:-3] # appended --port18124 and --leased
            for flag in ['--name','--out']:restored[restored.index(flag)+1]=serve.value(plan['server'],flag)
            self.assertEqual(restored,plan['server'])
            with patch.object(serve,'sha',return_value=serve.SCALE),patch.object(serve.Path,'read_text',return_value='research'),patch.object(serve,'read',side_effect=lambda p: json.loads(Path(p).read_bytes()) if Path(p).name == 'Mounts.json' else {'Cmd':['--max-model-len','100000'],'Env':[]}):
                with self.assertRaises(RuntimeError):serve.verify_plan(plan,200000)

    def test_scale_unique_rank_layer_coverage(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);inputs=root/'inputs.json';inputs.write_text(json.dumps({'mtp_scale_layer':'mtp.layers.0.self_attn.attn'}))
            names=['language_model.model.layers.'+str(i)+'.self_attn.attn' for i in range(3,64,4)]+['mtp.layers.0.self_attn.attn']
            for rank in [0,1]:
                (root/f'kv-load-{rank}.json').write_text(json.dumps({n:{'rank':rank,'artifact_sha256':serve.SCALE,'query_quantized':False,'kv_dtype':'fp8_e4m3'} for n in names}))
            with patch.object(serve,'INPUTS',inputs):
                serve.scale_receipts(root)
                bad=json.loads((root/'kv-load-1.json').read_text())
                for v in bad.values():v['rank']=0
                (root/'kv-load-1.json').write_text(json.dumps(bad))
                with self.assertRaisesRegex(RuntimeError,'coverage'):serve.scale_receipts(root)

    def test_legacy_qualification_cannot_drop_limitation(self):
        with self.assertRaisesRegex(RuntimeError,'limitation'):
            serve.validate_qualification({}, {'backend_configuration_qualified':True,'scope':serve.SCOPE,'unrestricted_model_quality_qualified':True})

    def test_readiness_and_unowned_socket(self):
        with patch.object(serve.Path,'read_text',return_value='PPid:\t123\n'):
            self.assertTrue(serve.readiness_owner({'lease_owner_pid':123,'owner_pid':456},123))
            self.assertFalse(serve.readiness_owner({'lease_owner_pid':122,'owner_pid':456},123))
            self.assertFalse(serve.readiness_owner({'lease_owner_pid':124,'owner_pid':456},124))
        with patch.object(serve.Path,'iterdir',return_value=iter([])),patch.object(serve.Path,'read_text',return_value='header\n0: 00000000:46A0 00000000:0000 0A 0:0 00:0 0 0 0 12345\n'):
            self.assertFalse(serve.owns_frontdoor(456))
        with self.assertRaises(RuntimeError):serve.stop_result(Path('/unrelated'))

    def test_actual200k_template_scope_and_trace(self):
        path=serve.ROOT/'results/bang_recurrence_testing_20260910/native5802-backend200k/plan.template.json'
        plan=serve.read(path)
        names,output=serve.context_scope(200000)
        self.assertEqual([j['name'] for j in plan['jobs']],names)
        self.assertEqual(Path(serve.value(plan['jobs'][-1]['command'],'--out')).name,output)
        serve.verify_plan(plan,200000)
        with self.assertRaises(RuntimeError):serve.context_scope(100001)

    def test_missing_and_wrong_inherited_leases(self):
        with patch.object(serve.os,'fstat',side_effect=OSError('closed')):
            with self.assertRaisesRegex(RuntimeError,'both inherited'):serve.verify_pair_lease()
        opened=SimpleNamespace(st_dev=1,st_ino=2);disk=SimpleNamespace(st_dev=1,st_ino=3)
        with patch.object(serve.os,'fstat',return_value=opened),patch.object(serve.Path,'stat',return_value=disk),patch.object(serve.Path,'read_text',return_value='lock: FLOCK ADVISORY WRITE'):
            with self.assertRaisesRegex(RuntimeError,'wrong inherited'):serve.verify_pair_lease()
        with patch.object(serve.os,'fstat',return_value=opened),patch.object(serve.Path,'stat',return_value=opened),patch.object(serve.Path,'read_text',return_value='pos: 0'):
            with self.assertRaisesRegex(RuntimeError,'does not hold'):serve.verify_pair_lease()

    def test_actual_gpu_run_with_temporary_cpu_only_locks(self):
        with tempfile.TemporaryDirectory() as d:
            code='import sys;from pathlib import Path;sys.path.insert(0,'+repr(str(Path(serve.__file__).parent))+');import serve;serve.ROOT=Path('+repr(d)+');serve.verify_pair_lease()'
            env=dict(os.environ,B70_GPU_LOCK=str(Path(d)/'gpu.lock'))
            result=subprocess.run([str(serve.REPO/'bin/gpu-run'),sys.executable,'-c',code],env=env,capture_output=True,text=True,timeout=10)
            self.assertEqual(result.returncode,0,result.stderr)

    def test_existing_frontdoor_auth_source_without_listener(self):
        source=(serve.REPO/'vllm/fp8/openai_key_frontdoor.py').read_text()
        cls=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='Handler')
        fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_authorized')
        ns={'hmac':hmac,'API_KEY':'temporary-fixture-key'}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'frontdoor-auth-source','exec'),ns)
        for headers,expected in [({'Authorization':'Bearer temporary-fixture-key'},True),({'X-API-Key':'temporary-fixture-key'},True),({'Authorization':'Bearer wrong'},False),({},False)]:
            self.assertEqual(ns['_authorized'](SimpleNamespace(headers=headers)),expected)

    def test_absence_guard_rejects_daemon_error(self):
        class R:pass
        r=R();r.returncode=1;r.stderr='error: no such object: owned'
        with patch.object(serve.subprocess,'run',return_value=r):serve.require_absent('owned')
        for message in ['Cannot connect to Docker daemon', 'dial unix /var/run/docker.sock: connect: no such file or directory']:
            r.stderr=message
            with patch.object(serve.subprocess,'run',return_value=r):
                with self.assertRaises(RuntimeError):serve.require_absent('owned')

    def test_actual_receipt_files_and_failed_semantic_gate(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);plan=self.plan(root);run=root/'run';run.mkdir();(run/'jobs').mkdir()
            health=root/'health.sh';health.write_text('strict fixture')
            plan['server'] += ['--health-probe',str(health)]
            cfg=serve.read(root/'Config.json');cfg['Cmd'] += ['--shutdown-timeout','30']
            (root/'Config.json').write_text(json.dumps(cfg))
            quality=run/'deterministic-round1';quality.mkdir()
            plan['out']=str(run);plan['jobs']=[{'name':'04-deterministic-round1-quality','command':['python3','/fixture/gate.py','agent',str(quality)]},{'name':'99-startup-host-trace-review','command':['python3','/fixture/trace.py']}]
            path=root/'plan.json';path.write_text(json.dumps(plan));path.with_suffix('.lifecycle-rc').write_text('0')
            (run/'WORKLOADS_PASSED').touch();(run/'exit.rc').write_text('0')
            (run/'arm-plan.json').write_text(json.dumps(plan))
            (run/'arm-results.json').write_text(json.dumps({j['name']:0 for j in plan['jobs']}))
            for job in plan['jobs']:(run/'jobs'/(job['name']+'.done')).write_text('0')
            (run/'manifest.json').write_text(json.dumps({'image':serve.IMAGE,'command':cfg['Cmd']+['--served-model-name','hotschmoe-dd','research'],
                'source_sha256':{'kv_campaign_server.py':serve.sha(serve.REPO/'vllm/fp8/kv_campaign_server.py')},'health_probe_sha256':serve.sha(health),
                'args':{'mtp':3,'p2p':0,'tensor_parallel_size':2,'prefix_off':False,'eager':False,'kv_dtype':'fp8_e4m3'}}))
            for stage in ['pre','post']:
                (run/(stage+'-xpu-health.log')).write_text('card 0: OK\ncard 1: OK\nHEALTHY')
                (run/(stage+'-xpu-collective-health.log')).write_text('COLLECTIVE_HEALTH_OK world_size=2\nxpu-collective-health: HEALTHY')
            (run/'server.log').write_text('Graph capturing finished')
            (run/'tp-host-review.json').write_text(json.dumps({'passed':True,'issues':[],'profiles':[{'matched':True}]}))
            (run/'models.json').write_text(json.dumps({'data':[{'id':'hotschmoe-dd'},{'id':'research'}]}))
            names=['language_model.model.layers.'+str(i)+'.self_attn.attn' for i in range(3,64,4)]+['mtp.layers.0.self_attn.attn']
            for rank in [0,1]:(run/f'kv-load-{rank}.json').write_text(json.dumps({name:{'rank':rank,'artifact_sha256':serve.SCALE,'kv_dtype':'fp8_e4m3','query_quantized':False} for name in names}))
            for mode in ['retrieval','reuse','guide','cancel','recovery']:
                (run/mode).mkdir(exist_ok=True);(run/mode/'summary.json').write_text(json.dumps({'passed':True,'mode':mode,'checks':1}))
            (run/'cancel/cancellation-drain.json').write_text(json.dumps({'passed':True}))
            gate=quality/'strict-quality-gate.json';gate.write_text(json.dumps({'passed':True,'checks':32}))
            with patch.object(serve,'require_absent'):
                hashes=serve.evidence(path);self.assertIn(str(gate),hashes)
                gate.write_text(json.dumps({'passed':False,'checks':32}))
                with self.assertRaisesRegex(RuntimeError,'semantic quality'):serve.evidence(path)

    def lifecycle(self,backend_rc):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);results=root/'results';results.mkdir();(root/'secrets').mkdir();(root/'secrets/dd_api_key').write_text('temporary-test-key')
            inputs=root/'inputs.json';inputs.write_text('{}');qual=root/'qualification.json';qual.write_text('{}')
            events=[];handlers={};state={};plan={'server':['backend','--out','old','--name','old','--served-alias','research']}
            class Proc:
                def __init__(self,kind):self.kind=kind;self.pid=55 if kind=='backend' else 56
                def poll(self):return None
                def terminate(self):events.append('front-terminate')
                def wait(self,timeout=None):events.append(self.kind+'-wait');return backend_rc if self.kind=='backend' else 0
            def popen(cmd,**kwargs):
                kwargs['stdout'].close()
                if cmd[0]=='backend':
                    server=Path(serve.value(cmd,'--out'));server.mkdir(parents=True);(server/'READY').touch();(server/'jobs').mkdir();state['server']=server
                    return Proc('backend')
                state['front']=True;return Proc('front')
            def sleep(_):
                if state.get('front'):handlers[signal.SIGTERM]()
            with patch.multiple(serve,ROOT=root,RESULTS=results,CURRENT=root/'current',INPUTS=inputs,QUAL=qual),patch.object(serve,'verify_inputs',return_value=plan),patch.object(serve,'validate_qualification'),patch.object(serve,'scale_receipts'),patch.object(serve,'verify_pair_lease'),patch.object(serve,'startup_jobs',return_value=[]),patch.object(serve,'owns_frontdoor',return_value=True),patch.object(serve.subprocess,'Popen',side_effect=popen),patch.object(serve.subprocess,'check_output',return_value=json.dumps([{'Image':serve.IMAGE}])),patch.object(serve.urllib.request,'urlopen',return_value=io.BytesIO(b'{"data":[{"id":"hotschmoe-dd"},{"id":"research"}]}')),patch.object(serve.signal,'signal',side_effect=lambda sig,fn:handlers.update({sig:fn})),patch.object(serve.time,'sleep',side_effect=sleep),patch.object(serve.sys,'argv',['serve.py','start','--leased']):
                if backend_rc:
                    with self.assertRaisesRegex(RuntimeError,'teardown failed'):serve.main()
                else:serve.main()
            self.assertTrue((state['server']/'STOP').exists())
            self.assertEqual(events,['front-terminate','front-wait','backend-wait'])
            self.assertEqual((state['server'].parent/'exit.rc').read_text().strip(),str(backend_rc))
            self.assertEqual(set(handlers),{signal.SIGTERM,signal.SIGINT,signal.SIGHUP})

    def test_signal_teardown_waits_backend(self):self.lifecycle(0)
    def test_failed_backend_teardown_preserved(self):self.lifecycle(7)

if __name__=='__main__':unittest.main()
