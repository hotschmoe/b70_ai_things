#!/usr/bin/env python3
"""Synthetic owned PLE and CPU actual-source hash controls; no model payloads."""
import copy,json,re,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from ple_owned_history_storage_v1 import OwnedPleHistory,PleHashConsts,ngram_rows
from original_gguf_vector_decoder_v2 import decode
from original_math_scalar import ple_dilated_conv

SOURCE=Path('/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261009T161219Z-n_v8o9lp/source/src/kernels/ngram.cpp')

class ToyOriginalRows:
 def __init__(self):
  self.reader=SimpleNamespace(tensors={name:(None,{'type':kind,'shape_ggml_order':list(shape)},None) for name,(kind,shape) in OwnedPleHistory.ROLES.items()});self.lookups=[]
 def rows(self,name,indices):
  ids=list(indices)
  if name=='per_layer_token_embd.weight':
   self.lookups.append(ids);result=[]
   for row in ids:
    raw=bytearray()
    for block in range(5):raw+=np.array([.001*(row%13+1)],dtype='<f2').tobytes()+bytes([((row+block+i)&15)|(((row+block+i+3)&15)<<4) for i in range(16)])
    result.append(decode(bytes(raw),'IQ4_NL'))
   return np.stack(result)
  if 'ple_norm_' in name:return np.ones((len(ids),10240),dtype=np.float64)
  if name.endswith('ple_conv1d.weight'):return np.tile(np.array([.25,.125,-.0625,.03125]),(len(ids),1))
  raise AssertionError('Toy projection provided analytically, no dense model array')

class ToyOwnedPle(OwnedPleHistory):
 def project(self,role,embedding):
  # Actual synthetic encoded-real Q8_0 row: constant d*code in each2560
  # columns, compressed analytically to avoid allocating fake200MiB weights.
  count=10240 if role=='ple_key.weight' else 2560;codes=np.arange(count)%63-31;scale=float(np.float16(.000977));return (codes*scale*np.sum(embedding)).astype('<f4').astype(np.float64)

class PleTests(unittest.TestCase):
 def setUp(self):self.p=ToyOriginalRows();self.s=ToyOwnedPle(self.p,'a'*64)
 def residual(self,index):return np.random.default_rng(500+index).normal(0,.2,(4,2560)).astype('<f4')
 def test_lookup_owned_headslowest_roworder(self):
  out=self.s.advance(19,self.residual(0));want=ngram_rows(19,[-1,-1]);self.assertEqual(out['row_ids_owned'],want);self.assertEqual(self.p.lookups[0],want);self.assertEqual(out['last_two_owned'],[-1,19]);self.assertEqual(out['embedding_F32'].shape,(2560,));self.assertFalse(out['full_model_math_qualified'])
 def test_token_zero_real_not_missing(self):self.assertNotEqual(ngram_rows(5,[-1,0]),ngram_rows(5,[-1,-1]))
 def test_EOS_cut_older_and_currentEOS_no_self_cut(self):
  eos=PleHashConsts().eos;self.assertEqual(ngram_rows(17,[88,eos]),ngram_rows(17,[99,eos]));self.assertNotEqual(ngram_rows(eos,[88,19]),ngram_rows(eos,[-1,-1]));self.assertNotEqual(ngram_rows(17,[eos,19]),ngram_rows(17,[88,19]))
 def test_oldest_first_window_order_matters(self):self.assertNotEqual(ngram_rows(17,[88,19]),ngram_rows(17,[19,88]))
 def test_uint64_wrap_XOR_modulo_not_rivals(self):
  c=PleHashConsts();token=0x7fffffff;prev=[0x7ffffffe,0x7ffffffd];got=ngram_rows(token,prev);context=[token,prev[1],prev[0]];products=[context[i]*c.multipliers[i] for i in range(3)];wrapped=(products[0]&((1<<64)-1))^(products[1]&((1<<64)-1));self.assertEqual(got[0],wrapped%c.vocab[0]);self.assertNotEqual(got[0],sum(products[:2])%c.vocab[0]);self.assertNotEqual(got[0],wrapped&(c.vocab[0]-1));self.assertNotEqual(got[0],(products[0]^products[1])%c.vocab[0])
 def test_dilation3_9rows_not_adjacent_taps(self):
  history=np.arange(9,dtype=float).reshape(9,1);new,conv=ple_dilated_conv(history,np.array([9.]),np.ones((4,1)));self.assertEqual(conv[0],18.);self.assertEqual(new[:,0].tolist(),list(range(1,10)))
 def test_owned_history_uses_normalized_previous_rows(self):
  a=self.s.advance(19,self.residual(0));b=self.s.advance(22,self.residual(1));self.assertEqual(np.count_nonzero(a['incoming_history_owned']),0);self.assertTrue(np.array_equal(b['incoming_history_owned'][-1],a['postprojection']['normalized'].reshape(-1)));self.assertEqual(b['predecessors_oldest_first'],[-1,19])
 def test_main_slot_checkpoint_replay_and_suffix(self):
  for i in range(10):self.s.advance(19+i,self.residual(i))
  snapshot=self.s.checkpoint();slot=ToyOwnedPle(self.p,'a'*64);slot.restore(snapshot,list(range(19,29)));a=self.s.advance(50,self.residual(10));b=slot.advance(50,self.residual(10));self.assertTrue(np.array_equal(a['postprojection']['result'],b['postprojection']['result']));self.assertEqual(self.s.digest(),slot.digest())
 def test_source_token_order_input_history_negatives(self):
  for i in range(4):self.s.advance(19+i,self.residual(i))
  snapshot=self.s.checkpoint();other=ToyOwnedPle(self.p,'b'*64)
  with self.assertRaises(ValueError):other.restore(snapshot,list(range(19,23)))
  other=ToyOwnedPle(self.p,'a'*64)
  with self.assertRaises(ValueError):other.restore(snapshot,list(reversed(range(19,23))))
  for field in ('history','last_two','records'):
   bad=copy.deepcopy(snapshot)
   if field=='history':bad[field].flat[0]+=.01
   elif field=='last_two':bad[field][0]+=1
   else:bad[field][0]['residual'].flat[0]+=.1
   with self.assertRaises(ValueError):other.restore(bad,list(range(19,23)))
 def test_reset_independent_sessions(self):
  a=self.s.advance(19,self.residual(0));self.s.advance(23,self.residual(1));self.s.reset();b=self.s.advance(19,self.residual(0));self.assertTrue(np.array_equal(a['postprojection']['result'],b['postprojection']['result']));self.assertEqual(self.s.last_two,[-1,19])
 def test_role_type_original_F32conv_not_stale_F16(self):
  self.p.reader.tensors['blk.1.ple_conv1d.weight'][1]['type']='F16'
  with self.assertRaises(ValueError):OwnedPleHistory(self.p,'a'*64)
 def test_nonfinite_input_refused(self):
  residual=self.residual(0);residual[0,0]=np.nan
  with self.assertRaises(ValueError):self.s.advance(19,residual)
 def test_scope_flags_no_native_or_fullmodel_or_gate(self):
  flags=self.s.qualification();self.assertFalse(flags['captured_state_or_row_ids_used']);self.assertFalse(flags['full_model_math_qualified']);self.assertFalse(flags['native_intrinsic_quant_or_reductions_emulated']);self.assertIsNone(flags['tolerance_gate'])
 def test_header_inventory_constants_and_roles_match(self):
  path=Path('/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f04-20261009/current-gguf-inventory.json');inventory=json.loads(path.read_text());files=[f for f in inventory['files'] if '/UD-Q4_K_XL/' in f['path']];metadata=files[0]['metadata'];c=PleHashConsts()
  for key,value in [('qwen4exp.ple.layer_multipliers',c.multipliers),('qwen4exp.ple.head_vocab_sizes',c.vocab),('qwen4exp.ple.head_offsets',c.offsets)]:self.assertEqual(tuple(metadata[key]['value']['values']),value)
  tensors={t['name']:t for file in files for t in file['tensors']}
  for name,(kind,shape) in OwnedPleHistory.ROLES.items():self.assertEqual((tensors[name]['type'],tuple(tensors[name]['shape_ggml_order'])),(kind,shape))
 def test_frozen_hash_oracle_vectors_match(self):
  path=SOURCE.with_name('ple_oracle_vectors.inc');lines=[]
  # Read only the first five HASH sections; do not inspect model table fixtures.
  with path.open() as source:
   active=False
   for line in source:
    lines.append(line)
    if 'case4_rows[' in line:active=True
    if active and line.strip()=='};':break
  text=''.join(lines)
  for number in range(5):
   def array(suffix):
    match=re.search(r'case%d_%s\[\d+\] = \{([^}]+)\}'%(number,suffix),text,re.S)
    return [int(value) for value in re.findall(r'-?\d+',match.group(1))]
   tokens=array('tokens');previous=array('prev');want=array('rows');got=[]
   for index,token in enumerate(tokens):got.extend(ngram_rows(token,previous[index*2:index*2+2]))
   self.assertEqual(got,want)

if __name__=='__main__':unittest.main()
