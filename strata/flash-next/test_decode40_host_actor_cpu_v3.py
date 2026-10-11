import io,json,queue,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
import run_decode40_host_profile_v3 as a

class Recording(io.BytesIO):
    def close(self):self.saved=self.getvalue();super().close()

class Controls(unittest.TestCase):
    def requests(self):return [{'ids':[1,2],'fresh':1,'max_new':64}for _ in range(4)]
    def test_actual_protocol_orders_ready_done_quit(self):
        lines=queue.Queue();lines.put('startup');lines.put('READY 1')
        for _ in range(4):lines.put('T 16');lines.put('DONE 1 2 12.5 20.5 stop')
        stream=Recording();rows=a.protocol(stream,lines,self.requests(),[False],lambda:None,time.monotonic()+2)
        self.assertEqual(stream.saved.count(b'GEN 64 seed=1 fresh=1 1,2\n'),4);self.assertTrue(stream.saved.endswith(b'QUIT\n'))
        self.assertEqual(rows[0]['output_ids'],[16]);self.assertEqual(rows[0]['done_signature'],['DONE','1','2','stop'])
    def test_error_count_quota_and_interrupt_fail(self):
        for bad in ('ERR actual failure','DONE 2 2 1 1 stop','DONE 0 2 nan 1 stop'):
            lines=queue.Queue();lines.put('READY');lines.put(bad)
            with self.assertRaises(ValueError):a.protocol(Recording(),lines,self.requests(),[False],lambda:None,time.monotonic()+1)
        with self.assertRaises(ValueError):a.protocol(Recording(),queue.Queue(),self.requests(),[True],lambda:None,time.monotonic()+1)
    def test_authoritative_early_eof_fails_without_deadline_wait(self):
        lines=queue.Queue();lines.put('READY');stream=Recording();start=time.monotonic()
        def guard():raise ValueError('Actual native CLI/pipe terminated before READY or DONE')
        with self.assertRaisesRegex(ValueError,'Actual native CLI/pipe terminated'):
            a.protocol(stream,lines,self.requests(),[False],guard,time.monotonic()+5)
        self.assertLess(time.monotonic()-start,1)
    def test_real_pump_continues_after_emit_error_and_quota(self):
        with tempfile.TemporaryDirectory()as t:
            def fail(line):raise RuntimeError('original queue callback')
            pump=a.Pump(io.BytesIO(b'one\ntwo\n'),Path(t)/'a',fail);proof=pump.finish()
            self.assertTrue(proof['eof']);self.assertFalse(proof['passed']);self.assertTrue(proof['errors'])
            with patch.object(a,'MAX_PIPE',3):
                pump=a.Pump(io.BytesIO(b'one\ntwo\n'),Path(t)/'b',lambda _:None);proof=pump.finish()
                self.assertTrue(proof['eof']);self.assertFalse(proof['passed']);self.assertEqual(proof['bytes'],8)
    def metadata_object(self):
        cmd=['docker','run','--name','b70-h36-tokenizer-12','--entrypoint','/usr/bin/env','--user','1000:1000','--workdir','/','-v','tiny:/tokenizer:ro','sha256:abc','-i','real']
        obj={'RestartCount':0,'Name':'/b70-h36-tokenizer-12','Config':{'Image':'sha256:abc','Labels':{'b70.decode40.host':'binding'},'Cmd':['-i','real'],'Entrypoint':['/usr/bin/env'],'Env':[],'User':'1000:1000','WorkingDir':'/'},'HostConfig':{'AutoRemove':False,'Binds':['tiny:/tokenizer:ro'],'Privileged':False,'NetworkMode':'none','Memory':512<<20,'MemorySwap':512<<20,'ReadonlyRootfs':True,'NanoCpus':1000000000,'PidsLimit':128,'Devices':[],'DeviceRequests':[],'GroupAdd':None}}
        return cmd,obj
    def test_exact_owned_recipe_and_foreign_refusal(self):
        cmd,obj=self.metadata_object()
        with patch.object(a.c,'inspected',return_value=obj):self.assertEqual(a.inspect_owned('b70-h36-tokenizer-12',cmd,'sha256:abc','binding'),obj)
        obj['Config']['Cmd']=['FOREIGN']
        with patch.object(a.c,'inspected',return_value=obj),self.assertRaises(ValueError):a.inspect_owned('b70-h36-tokenizer-12',cmd,'sha256:abc','binding')
    def test_sorted_mount_bijection_and_resources(self):
        cmd,obj=self.metadata_object();cmd[cmd.index('sha256:abc'):cmd.index('sha256:abc')]=['-v','tiny2:/second:ro'];obj['HostConfig']['Binds']=['tiny2:/second:ro','tiny:/tokenizer:ro']
        a.contract.inspection(obj,'b70-h36-tokenizer-12',cmd,'sha256:abc','binding',True)
        for field,value in [('Privileged',True),('Memory',1),('NanoCpus',1000000000.0),('Devices',[{'foreign':True}])]:
            old=obj['HostConfig'][field];obj['HostConfig'][field]=value
            with self.assertRaises(ValueError):a.contract.inspection(obj,'b70-h36-tokenizer-12',cmd,'sha256:abc','binding',True)
            obj['HostConfig'][field]=old
        with self.assertRaises(ValueError):a.contract.mount_binding(['a:/same:ro','b:/same:ro'],['a:/same:ro','b:/same:ro'])
    def test_exact_restart_and_pids_types(self):
        cmd,obj=self.metadata_object()
        for key,value in [('RestartCount',0.0),('RestartCount',1)]:
            obj[key]=value
            with self.assertRaises(ValueError):a.contract.inspection(obj,'b70-h36-tokenizer-12',cmd,'sha256:abc','binding',True)
        obj['RestartCount']=0;obj['HostConfig']['PidsLimit']=128.0
        with self.assertRaises(ValueError):a.contract.inspection(obj,'b70-h36-tokenizer-12',cmd,'sha256:abc','binding',True)
    def test_noncreated_failed_cli_absence_is_not_terminal_pass(self):
        proc=type('Proc',(),{'poll':lambda _:1,'returncode':1})()
        with patch.object(a.c,'absent',return_value=True):proof=a.retire('owned',[],'image','binding',proc,True)
        self.assertTrue(proof['container_absent']);self.assertIsNone(proof['State'])
        with self.assertRaises(ValueError):a.terminal(proof['State'])
    def test_successful_rm_unknown_retries_absence_only(self):
        obj={'Id':'exact'}
        states=iter([False,False,True])
        with tempfile.TemporaryDirectory()as t,patch.object(a.c,'absent',side_effect=lambda _:next(states)),patch.object(a,'inspect_owned',return_value={'Id':'exact','State':{'Running':False}}),patch.object(a.subprocess,'run')as run,patch.object(a.time,'sleep'):
            row=a.remove_owned(Path(t),obj,'owned',[],'image','binding');self.assertTrue(row['removed']);self.assertEqual(run.call_count,1);self.assertEqual(run.call_args.args[0],['docker','rm','exact'])
    def test_original_speed_join_requires_admitted_source_and_ids(self):
        with tempfile.TemporaryDirectory()as t:
            root=Path(t);path=root/'engine-token-trace.jsonl';ids={'short':[1],'prefill_about512':[2]}
            path.write_text('\n'.join(json.dumps({'kind':'engine_begin','submitted_ids':i,'rendered_matches_submitted':True})for i in ids.values())+'\n')
            m={'engine_receipt_sha256':'engine','executable_sha256':'elf'}
            (root/'prepared.json').write_text(json.dumps(m));(root/'speed-parent.json').write_text(json.dumps({'passed':True}))
            self.assertTrue(a.match_speed_inputs(path,ids,m)['independent_tokenizer_matches_actual_speed_inputs'])
            with self.assertRaises(ValueError):a.match_speed_inputs(path,{'short':[9],'prefill_about512':[2]},m)
            (root/'speed-parent.json').write_text(json.dumps({'passed':False}))
            with self.assertRaises(ValueError):a.match_speed_inputs(path,ids,m)
    def test_typed_normal_terminal(self):
        state={'Running':False,'OOMKilled':False,'ExitCode':0,'Error':'','Status':'exited','Paused':False,'Restarting':False,'Dead':False,'Pid':0};a.terminal(state)
        for key,value in [('ExitCode',False),('OOMKilled',True),('Error','contradiction'),('Status','running')]:
            bad=dict(state);bad[key]=value
            with self.assertRaises(ValueError):a.terminal(bad)

    def test_real_missing_pump_setup_streams_retire(self):
        import subprocess,sys
        with tempfile.TemporaryDirectory()as t:
            proc=subprocess.Popen([sys.executable,'-c','import sys;print("tiny");print("err",file=sys.stderr)'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            with patch.object(a.c,'absent',return_value=True):
                row=a.setup_failure_drain(Path(t),'owned',[],'image','binding',proc,[('stdout',proc.stdout),('stderr',proc.stderr)],'Synthetic Pump setup failure')
            self.assertTrue(row['actual_missing_streams_drained']);self.assertEqual(row['original_CLI_return_code'],0);self.assertIsNotNone(proc.poll())
            self.assertTrue(proc.stdout.closed and proc.stderr.closed)

if __name__=='__main__':unittest.main()
