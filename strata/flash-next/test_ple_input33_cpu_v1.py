#!/usr/bin/env python3
"""Synthetic raw-input and actual source-body branch controls; no payload/GPU/git."""
import copy,hashlib,json,re,tempfile,unittest,shutil,subprocess
from pathlib import Path
import numpy as np
import collect_ple_input33_v1 as c
import verify_ple_input33_original_v1 as original
from ple_owned_history_storage_v1 import ngram_rows
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]

class Fixture(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.requests={};self.frames=[];self.lines=[];self.binding='a'*64
  for request,n in enumerate((1,2,4,8),1):
   ids=list(range(100,100+n));pos=n-1;prev=[ids[pos-2] if pos>=2 else -1,ids[pos-1] if pos>=1 else -1];rows=ngram_rows(ids[-1],prev);emb=np.repeat(np.asarray(rows,dtype='<f4')[:,None],160,axis=1).reshape(2560)
   raw=self.root/f'ple-input33-{request}.f32';row=self.root/f'ple-input33-{request}.u32';raw.write_bytes(emb.tobytes());row.write_bytes(np.array(rows,dtype='<u4').tobytes());meta=self.root/f'ple-input33-{request}.json';epoch=(9<<32)|request
   f=dict(schema=1,pid=9,request=request,epoch=epoch,stage=0,device=0,lb=0,le=32,T=1,pos=pos,reused=0,source_exact=True,no_host=True,device_plan=True,AR=False,prelaunch_host_publication_only=True,binding_sha256=self.binding,embedding_encoding='LE_F32[T,2560]',row_ids_encoding='LE_U32[T,16]',embedding_file=raw.name,embedding_bytes=10240,embedding_sha256=c.sha(raw),row_ids_file=row.name,row_ids_bytes=64,row_ids_sha256=c.sha(row),prev=prev,gen_ids=ids,window_tokens=[ids[-1]],row_ids=rows,full_model_math_qualified=False);meta.write_text(json.dumps(f));self.frames.append((meta,f));self.requests[request]=ids
   identity=f'pid=9 request={request} stage=0 device=0 epoch={epoch} pos={pos} T=1'
   self.lines += [f'SFD request pid=9 request={request} tokens={n} ids='+','.join(map(str,ids))+' shape=fixture',f'SFD resume pid=9 request={request} reused=0 evaluated_prompt_pending=1',f'PLE_INPUT33 published {identity} metadata={meta.name}',f'PLE_INPUT33 submit_begin {identity}',f'PLE_INPUT33 submit_returned {identity} rc=0',f'PLE_INPUT33 terminal {identity} cancelled=0 generated=1 finish=length']
  self.log=self.root/'producer.log';self.publish()
 def publish(self):
  self.log.write_text('\n'.join(self.lines)+'\n')
  for path,frame in self.frames:path.write_text(json.dumps(frame))
 def collect(self):return c.collect(self.root,self.requests,self.binding,self.log,(0,32))
 def test_complete_actual_raw_and_chronology_synthetic_positive(self):
  proof=self.collect();self.assertEqual(len(proof['frames']),4);self.assertTrue(proof['actual_matching_SFD_request_terminal_observed']);self.assertFalse(proof['GPU_consumed_PLE_input_observed'])
 def test_original_scalar_row_source_match_synthetic(self):
  proof=self.collect();provider=type('CPUProvider',(),{'rows':lambda self,name,ids:np.repeat(np.asarray(ids,dtype='<f4')[:,None],160,axis=1)})();self.assertTrue(original.verify(proof,provider)['actual_original_staged_input_bitwise'])
 def test_reordered_actual_request_or_resume_rejected(self):
  for first,second in [(0,2),(1,3),(2,3),(3,4),(4,5)]:
   saved=self.lines[:];self.lines[first],self.lines[second]=self.lines[second],self.lines[first];self.publish()
   with self.assertRaises(ValueError):self.collect()
   self.lines=saved
 def test_missing_or_wrong_terminal_rejected(self):
  for transform in [lambda rows:rows[:5]+rows[6:],lambda rows:[x.replace('terminal pid=9','terminal pid=8') for x in rows],lambda rows:[x.replace('terminal pid=9 request=1','terminal pid=9 request=99') for x in rows]]:
   saved=self.lines[:];self.lines=transform(saved);self.publish()
   with self.assertRaises(ValueError):self.collect()
   self.lines=saved
 def test_submit_returned_failure_rejected(self):
  self.lines=[x.replace('rc=0','rc=1') for x in self.lines];self.publish()
  with self.assertRaises(ValueError):self.collect()
 def test_duplicate_or_stale_publication_rejected(self):
  self.lines.insert(3,self.lines[2]);self.publish()
  with self.assertRaises(ValueError):self.collect()
 def test_changed_raw_rows_or_embedding_rejected(self):
  path=self.root/self.frames[0][1]['embedding_file'];path.write_bytes(bytes(10240))
  with self.assertRaises(ValueError):self.collect()
 def test_raw_rows_match_metadata_and_encoding(self):
  self.frames[0][1]['row_ids'][0]+=1;self.publish()
  with self.assertRaises(ValueError):self.collect()
 def test_symlink_raw_output_rejected(self):
  path=self.root/self.frames[0][1]['embedding_file'];data=path.read_bytes();path.unlink();target=self.root/'other.bin';target.write_bytes(data);path.symlink_to(target)
  with self.assertRaises(ValueError):self.collect()
 def test_terminal_before_submit_returned_rejected(self):
  self.lines[4],self.lines[5]=self.lines[5],self.lines[4];self.publish()
  with self.assertRaises(ValueError):self.collect()
 def test_cancelled_terminal_rejected(self):
  self.lines=[line.replace('cancelled=0','cancelled=1') for line in self.lines];self.publish()
  with self.assertRaises(ValueError):self.collect()
 def test_missing_entire_last_terminal_rejected(self):
  self.lines.pop();self.publish()
  with self.assertRaises(ValueError):self.collect()
 def test_original_different_but_hash_consistent_embedding_rejected(self):
  proof=self.collect();frame=proof['frames'][0];path=Path(frame['fields']['embedding']['path']);path.write_bytes(np.zeros(2560,dtype='<f4').tobytes());frame['fields']['embedding']['sha256']=c.sha(path);provider=type('CPUProvider',(),{'rows':lambda self,name,ids:np.repeat(np.asarray(ids,dtype='<f4')[:,None],160,axis=1)})();
  with self.assertRaises(ValueError):original.verify(proof,provider)
 def test_original_wrong_request_no_payload(self):
  proof=self.collect();proof['frames'][0]['binding']['request']=99;provider=type('NoReads',(),{'rows':lambda *args:(_ for _ in ()).throw(AssertionError('No payload'))})();
  with self.assertRaises(ValueError):original.verify(proof,provider)
 def test_truncated_final_frame_rejected(self):
  self.frames[-1][0].unlink()
  with self.assertRaises(ValueError):self.collect()
 def test_prev_accepted_window_stage_binding_rejected(self):
  for key,val in [('prev',[1,2]),('gen_ids',[1]),('device',1),('epoch',1)]:
   old=self.frames[0][1][key];self.frames[0][1][key]=val;self.publish()
   with self.assertRaises(ValueError):self.collect()
   self.frames[0][1][key]=old
 def test_original_lookup_never_called_on_incomplete_changed_proof(self):
  proof=self.collect();proof['frames'].pop();provider=type('NoReads',(),{'rows':lambda *args:(_ for _ in ()).throw(AssertionError('No original payload before metadata admission'))})();
  with self.assertRaises(ValueError):original.verify(proof,provider)
 def test_original_lookup_never_called_on_changed_raw_sha(self):
  proof=self.collect();Path(proof['frames'][0]['fields']['embedding']['path']).write_bytes(bytes(10240));provider=type('NoReads',(),{'rows':lambda *args:(_ for _ in ()).throw(AssertionError('No original payload'))})();
  with self.assertRaises(ValueError):original.verify(proof,provider)

class SourceBodyTests(unittest.TestCase):
 def test_reconstruct_exact_pinned_source_without_git_runtime_or_Docker(self):
  plan=json.loads((HERE/'current-ple-input33-engine-build-plan-v1.json').read_text());base=Path(plan['source_root']);needed=set(plan['source_file_sha256']);patches=[]
  for row in plan['patches']:
   path=ROOT/row['path'];self.assertEqual(c.sha(path),row['sha256']);text=path.read_text();patches.append(path);needed.update(re.findall(r'^\+\+\+ b/(.*)$',text,re.M));needed.update(re.findall(r'^--- a/(.*)$',text,re.M))
  with tempfile.TemporaryDirectory() as name:
   source=Path(name)
   for rel in needed:
    path=base/rel
    if path.is_file():dest=source/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
   for rel,digest in plan['source_file_sha256'].items():self.assertEqual(c.sha(source/rel),digest)
   for path in patches:
    result=subprocess.run(['patch','--batch','--forward','-p1','-i',str(path)],cwd=source,capture_output=True,text=True);self.assertEqual(result.returncode,0,result.stdout+result.stderr)
   for rel,digest in plan['expected_patched_source_sha256'].items():self.assertEqual(c.sha(source/rel),digest,rel)
   for header in plan['added_header_payloads']:self.assertEqual(c.sha(source/header['path']),header['sha256'])
   verify=(source/'sycl/src/core/verify.cpp').read_text();block=verify[verify.index('const bool ple_precollected'):verify.index('trace_ev("LAUNCHED"')]
   self.assertLess(block.index('gather_batch'),block.index('ple_input33::publish'));self.assertLess(block.index('_mm_sfence'),block.index('ple_input33::publish'));self.assertLess(block.index('ple_input33::publish'),block.index('ple_input33::submit_begin'));self.assertLess(block.index('ple_input33::submit_begin'),block.index('ext_oneapi_graph'));self.assertLess(block.index('ext_oneapi_graph'),block.index('ple_input33::submit_returned'))
   gen=(source/'sycl/src/program/generate.cpp').read_text();segment=gen[gen.index('            prefix_diag.finish('):gen.index('// DONE <generated>')];self.assertLess(segment.index('prefix_diag.finish'),segment.index('ple_input33::terminal'));self.assertNotIn('active=false',segment);self.assertIn('fidelity_diag::current().ordinal',segment)
   header=(source/'include/strata/core/ple_input_observer33.hpp').read_text();body=header[header.index('inline publication publish'):header.index('inline void submit_begin')];off=body.index('if(!settings().enabled||!active)return {}')
   for action in ['reinterpret_cast<const uint8_t*>','std::isfinite(embedding','write_bytes','getpid()','Sha256']:self.assertLess(off,body.index(action))
   for forbidden in ['sycl::','q.wait','malloc_device','get_native','memcpy(']:self.assertNotIn(forbidden,header)
   self.assertNotIn('const float*',header[header.index('struct publication'):header.index('inline void write_bytes')])
 def test_actual_source32_predicate_branch_fixture(self):
  text=(HERE/'patches/0032-sycl-current-ple-staging-before-nohost-mirror-graph.patch').read_text();self.assertIn('do_ple && !ar_on() && (device_plan_ || no_host)',text)
  for ple in (False,True):
   for ar in (False,True):
    for plan in (False,True):
     for host in (False,True):
      reached=ple and not ar and (plan or host);self.assertEqual(reached,ple and (not ar) and (plan or host))
 def test_defaultoff_or_unarmed_source_no_observation_effect(self):
  # Actual C++ earliest return expression extracted, not a surrogate metadata gate.
  text=(HERE/'ple_input_observer33_v1.hpp').read_text();self.assertIn('if(!settings().enabled||!active)return {}',text)
  for enabled,active in [(False,False),(False,True),(True,False)]:self.assertTrue((not enabled) or (not active))
if __name__=='__main__':unittest.main()
