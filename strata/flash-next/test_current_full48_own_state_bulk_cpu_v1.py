import unittest,tempfile,json,hashlib
from types import SimpleNamespace
from unittest.mock import patch
import current_full48_own_state_bulk_v1 as a
ARGS=['--pack','/pack','--native','/model','--ple-gguf','/ple','--expert-profile','/profile','--expert-cache','auto','--stream-experts','--no-prefill-borrow','--prefill','64','--spec','2','--max-context','2048','--kv','fp16','--ple-io','mmap','--ple-row-cache','65536','--pcie-frac','1','--vram-reserve-mib','2048','--prompt-cache','0','--suffix-draft','0','--lookup-chain','0','--adapt-swaps','0','--adapt-every','0','--batch','0']
ENV={'STRATA_SYCL_NATIVE_HC':'1','STRATA_VERIFY_DEVICE_PLAN':'1','STRATA_VERIFY_NO_HOST':'1','STRATA_STAGE_MIRRORS':'1','STRATA_STAGE_MIRROR_SEGMENT_MIB':'1024','STRATA_MIRROR_MIB':'65536','STRATA_QSA_INDEXER_MATERIALIZED_HALF':'1'}
class Controls(unittest.TestCase):
 def test_actual_current_cacheOFF_schedule(self):
  v=a.schedule([248045,8678,198,15666],ARGS,ENV);self.assertEqual([(w['position'],w['rows'],w['math_route'])for w in v['windows']],[(0,2,'verifier'),(2,1,'verifier'),(3,1,'verifier')]);self.assertTrue(v['fresh_zero_state_required'])
 def test_unmodeled_contracts_refused(self):
  for env in({**ENV,'STRATA_QSA_INDEXER_MATERIALIZED_HALF':'0'},{**ENV,'STRATA_SYCL_NATIVE_GROUP_F16':'1'},{**ENV,'STRATA_UNKNOWN_MATH':'1'}):
   with self.assertRaises(ValueError):a.schedule([248045],ARGS,env)
  bad=ARGS.copy();bad[bad.index('--prompt-cache')+1]='3'
  with self.assertRaises(ValueError):a.schedule([248045],bad,ENV)
 def test_completed_parity_without_closure_refused(self):
  with tempfile.TemporaryDirectory()as tmp:
   p=__import__('pathlib').Path(tmp)/'report.json';v={'passed':True,'errors':[],'postCPU_full4':{'passed':False},'bulk_vs_executed_scalar':{'all_bitwise_equal':True}};p.write_text(json.dumps(v));digest=hashlib.sha256(p.read_bytes()).hexdigest()
   with self.assertRaises(ValueError):a.first_HC_gate(p,digest,'b'*64)
 def test_missing_actual_firstHC_gate_precedes_provider(self):
  p=SimpleNamespace(rows=lambda *x:(_ for _ in()).throw(AssertionError('provider must not run')))
  with patch.object(a,'source_contract',side_effect=AssertionError('source should not run before gate')):
   with self.assertRaises(ValueError):a.CurrentFull48OwnStateBulk(p,'a'*64,ARGS,ENV,generation=47,rows_dot_elf='/tmp/elf',rows_dot_sha256='b'*64,first_HC_report='/tmp/missing-firstHC',first_HC_report_sha256='c'*64)
if __name__=='__main__':unittest.main()
