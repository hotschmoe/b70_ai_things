import hashlib,json,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
import validate_decode40_host_profile_v3 as v

class Controls(unittest.TestCase):
    def test_original_timestamps_protocol_and_mutation(self):
        with tempfile.TemporaryDirectory()as t:
            root=Path(t);raw=['READY']
            for _ in range(4):raw+=['T 16','DONE 1 2 1 2 stop']
            (root/'native.stdout.log').write_text('\n'.join(raw)+'\n')
            timed=[{'seq':i,'line':line,'monotonic_ns':100+i}for i,line in enumerate(raw)]
            (root/'native.stdout-times.jsonl').write_text('\n'.join(map(json.dumps,timed))+'\n')
            saved=[{'ids':[1,2],'fresh':1,'max_new':64,'output_ids':[16],'done':'DONE 1 2 1 2 stop','done_signature':['DONE','1','2','stop'],'native_pipe_token_ns':[101+2*i]}for i in range(4)]
            packets=['GEN 64 seed=1 fresh=1 1,2\n']*4+['QUIT\n']
            (root/'native.stdin.jsonl').write_text('\n'.join(json.dumps({'packet':p})for p in packets)+'\n')
            self.assertEqual(len(v.replay(root,saved)),4)
            timed[1]['monotonic_ns']=True;(root/'native.stdout-times.jsonl').write_text('\n'.join(map(json.dumps,timed))+'\n')
            with self.assertRaises(ValueError):v.replay(root,saved)
    def test_exact_health_command_saved_receipt_and_mutation(self):
        with tempfile.TemporaryDirectory()as t:
            root=Path(t);label='test';cmd=['tiny-original'];(root/(label+'.log')).write_text('HEALTHY\n');(root/(label+'.command.json')).write_text(json.dumps(cmd))
            row={'command':cmd,'passed':True,'return_code':0,'command_error':None,'phase_cleanup_error':None,'eof':True,'reader_retired':True,'error':None,'path':str(root/(label+'.log')),'sha256':v.c.sha(root/(label+'.log')),'command_sha256':v.c.sha(root/(label+'.command.json')),'started_command_epoch':1,'started_epoch':2,'completed_epoch':3,'finished_epoch':4}
            path=root/(label+'.receipt.json');path.write_text(json.dumps(row));self.assertEqual(v.command_row(root,label,cmd),row)
            row['return_code']=False;path.write_text(json.dumps(row))
            with self.assertRaises(ValueError):v.command_row(root,label,cmd)
    def test_exact_page_roster_source_scope(self):
        from source_page_watchdog_v3 import KNOWN_PAGES
        value={'epoch':1,'passed':True,'path':'/tiny/third','stat_before':[1,2,3,4,5],'stat_after':[1,2,3,4,5],'rows':[{'passed':True,'offset':offset,'bytes':4096,'sha256':sha,'expected_sha256':sha}for offset,sha in KNOWN_PAGES]}
        v.pages(value,Path('/tiny/third'),[1,2,3,4,5]);value['rows'][0]['sha256']='0'*64
        with self.assertRaises(ValueError):v.pages(value,Path('/tiny/third'),[1,2,3,4,5])
    def test_publisher_savedfile_body_currentstat_join(self):
        with tempfile.TemporaryDirectory()as t:
            root=Path(t);rows=[];lock={'revision':'tiny','destination':str(root),'files':[]}
            for i in range(4):
                path=root/('part'+str(i));path.write_bytes(b'x');sig=v.c.stat_signature(path);sha=v.c.sha(path)
                lock['files'].append({'path':'UD-Q4_K_XL/'+path.name,'sha256':sha,'size':1})
                rows.append({'path':str(root/'UD-Q4_K_XL'/path.name),'passed':True,'sha256':sha,'expected_sha256':sha,'bytes':1,'stat_before':sig,'stat_after':sig})
            report={'owned_model_terminal_epoch':10,'post_health':{'finished_epoch':11}}
            page={'passed':True,'path':rows[2]['path'],'stat_before':rows[2]['stat_before'],'stat_after':rows[2]['stat_after'],'rows':[{'passed':True,'offset':o,'bytes':4096,'sha256':s,'expected_sha256':s}for o,s in v.a.q.watchdog.KNOWN_PAGES]}
            for phase in('pre','post'):
                value={'passed':True,'lock_sha256':'tiny','model_revision':'tiny','rows':rows,'started':12,'finished':13,'after_terminal_and_post_health_epoch':11}
                (root/(phase+'-full4.json')).write_text(json.dumps(value));report[phase+'_full4']=value;report[phase+'_pages']={**page,'epoch':11};report[phase+'_after_pages']={**page,'epoch':14}
            original_read=v.c.read
            def read(path):return lock if Path(path).name=='model-lock.json'else original_read(path)
            with patch.object(v.c,'read',side_effect=read),patch.object(v.c,'sha',return_value='tiny'),patch.object(v.c,'stat_signature',side_effect=lambda p:rows[int(Path(p).name[-1])]['stat_after']),patch.object(v.a.q.watchdog,'guard'):
                v.source4(root,report)
                body=original_read(root/'post-full4.json');body['rows'][0]['bytes']=2;(root/'post-full4.json').write_text(json.dumps(body))
                with self.assertRaises(ValueError):v.source4(root,report)

    def test_late_prescan_cannot_admit_first_launch(self):
        report={'pre_full4':{'finished':1},'pre_after_pages':{'epoch':2},'OFF-pre_health':{'started_epoch':3}}
        v.pre_launch_binding(report,{'launch_epoch':4});report['pre_full4']['finished']=5
        with self.assertRaises(ValueError):v.pre_launch_binding(report,{'launch_epoch':4})
    def test_finite_typed_epochs(self):
        for value in(True,float('nan'),float('inf'),0,-1):
            with self.assertRaises(ValueError):v.epoch(value)
        self.assertEqual(v.epoch(1.5),1.5)
    def test_raw_missing_newline_refused_before_replay(self):
        with tempfile.TemporaryDirectory()as t:
            root=Path(t);(root/'native.stdout.log').write_bytes(b'READY')
            with self.assertRaisesRegex(ValueError,'Complete original native line'):v.replay(root,[])
if __name__=='__main__':unittest.main()
