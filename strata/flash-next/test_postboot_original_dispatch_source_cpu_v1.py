"""Actual tiny files; no model/SDK or helper actions."""
import copy,hashlib,tempfile,unittest
from pathlib import Path
from postboot_original_dispatch_source_v1 import admit,CODE,stat5
HERE=Path(__file__).resolve().parent

class Controls(unittest.TestCase):
    def fixture(self,root):
        files={}
        for name in CODE:
            p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'CPP source fixture\n')
            files[name]={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'stat5':stat5(p)}
        new={'files':{'unchanged':'source'},'HC_source':{'dispatch':{'files':files,'flag':False},'kernels':{'key':'value'}}}
        old=copy.deepcopy(new);device=next(iter(files.values()))['stat5'][0]
        for r in old['HC_source']['dispatch']['files'].values():r['stat5'][0]=device+1
        association={'mapping':[{'historical_stat5':[device+1,1,1,1,1],'current_stat5':[device,1,1,1,1]}]}
        return old,new,association
    def test_exact_source_bytes_old_current_fields_separate(self):
        with tempfile.TemporaryDirectory()as d:
            old,new,a=self.fixture(Path(d));before=copy.deepcopy(old)
            result=admit(old,new,a);self.assertEqual(len(result['rows']),3)
            self.assertEqual(old,before);self.assertEqual(result['original_binding'],old)
            self.assertEqual(result['current_binding'],new);self.assertFalse(result['old_current_source_gate_passed'])
            self.assertFalse(result['compiled_flags_or_mathematics_changed'])
    def test_all_nondevice_stats_and_hashes_refused(self):
        with tempfile.TemporaryDirectory()as d:
            old,new,a=self.fixture(Path(d))
            for index in range(1,5):
                bad=copy.deepcopy(new);bad['HC_source']['dispatch']['files'][CODE[0]]['stat5'][index]+=1
                self.assertRaises(ValueError,admit,old,bad,a)
            for field in ('path','sha256'):
                bad=copy.deepcopy(new);bad['HC_source']['dispatch']['files'][CODE[0]][field]='FOREIGN'
                self.assertRaises(ValueError,admit,old,bad,a)
    def test_nonstat_source_scope_and_bool_or_foreigndevice_refused(self):
        with tempfile.TemporaryDirectory()as d:
            old,new,a=self.fixture(Path(d))
            for change in ('flag','kernel','bool','device','extra'):
                bad=copy.deepcopy(new)
                if change=='flag':bad['HC_source']['dispatch']['flag']=True
                elif change=='kernel':bad['HC_source']['kernels']['key']='FOREIGN'
                elif change=='bool':bad['HC_source']['dispatch']['files'][CODE[0]]['stat5'][0]=True
                elif change=='device':bad['HC_source']['dispatch']['files'][CODE[0]]['stat5'][0]+=100
                else:bad['HC_source']['dispatch']['files']['extra']=bad['HC_source']['dispatch']['files'][CODE[0]]
                self.assertRaises(ValueError,admit,old,bad,a)
    def test_exact_work_successor_only_source_guard_changes(self):
        names=('postboot_original_rms_dependencies','postboot_owned_hc_work','postboot_owned_qsa_work',
               'postboot_rms37_v7_historical_reader','postboot_rms37_v8_historical_reader',
               'postboot_owned_hc_v3_historical_reader','postboot_owned_qsa_v4_historical_reader')
        for name in names:
            expected=(HERE/(name+'_v2.py')).read_text()
            for ref in names:expected=expected.replace(ref+'_v2',ref+'_v3')
            if name=='postboot_owned_hc_work':
                expected='from postboot_original_dispatch_source_v1 import admit as source_association\n'+expected
                expected=expected.replace("report['first_source_binding']==self.first_source_binding",
                    "source_association(report['first_source_binding'],self.first_source_binding,self.model_association)['only_declared_source_st_dev_associated'] is True")
            self.assertEqual((HERE/(name+'_v3.py')).read_text(),expected)

if __name__=='__main__':unittest.main()
