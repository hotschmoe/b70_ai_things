"""Small CPU tests of streamed tensor layout, head chunks and refusal gates."""
import unittest
import torch
from streamed import Weights


class FixtureWeights(Weights):
    def __init__(self,tensors):
        self.tensors=tensors;self.index={k:'fixture' for k in tensors}
        self.headers={k:tuple(v.shape) for k,v in tensors.items()}
    def read(self,name,sl=None):
        t=self.tensors[name]
        return t.clone() if sl is None else t[sl].clone()


class TestStreamed(unittest.TestCase):
    def test_chunked_gptq_and_unhandled(self):
        k,n=256,136
        values=(torch.arange(k*n).reshape(k,n)+torch.arange(k)[:,None])%16
        packed=torch.zeros(k//8,n,dtype=torch.int64)
        for j in range(8):packed|=values[j::8]<<(4*j)
        scale=torch.full((2,n),.0137,dtype=torch.float16)
        scale[1]*=3
        prefix='model.language_model.layers.0.proj.'
        w=FixtureWeights({prefix+'qweight':packed.int(),prefix+'qzeros':torch.zeros(2,n//8,dtype=torch.int32),prefix+'scales':scale})
        layer=torch.nn.Module();layer.proj=torch.nn.Linear(k,n,bias=False,device='meta')
        w.load_layer(layer,0)
        expected=((values.float()-8)*scale.float().repeat_interleave(128,0)).half().float().T
        self.assertTrue(torch.equal(layer.proj.weight,expected))
        w.index[prefix+'unknown']='fixture'
        with self.assertRaises(ValueError): w.mapping(layer,0)

    def test_head_chunks_and_sparse_embedding(self):
        torch.manual_seed(7)
        table=torch.randn(37,16).half()
        w=FixtureWeights({'lm_head.weight':table,'model.language_model.embed_tokens.weight':table})
        x=torch.randn(3,16)
        torch.testing.assert_close(w.head(x,chunk=7),x@table.float().T,atol=1e-5,rtol=1e-5)
        ids=torch.tensor([[3,1,3,36,0]])
        self.assertTrue(torch.equal(w.embedding(ids),table[ids].float()))
        with self.assertRaises(ValueError):w.head(torch.randn(17,16))

    def test_direct_dtype_and_causal_scoring(self):
        from run_reference import score_rows
        prefix='model.language_model.layers.0.'
        value=torch.tensor([1e-9],dtype=torch.bfloat16)
        w=FixtureWeights({prefix+'linear_attn.A_log':value,prefix+'norm.weight':value})
        layer=torch.nn.Module();layer.linear_attn=torch.nn.Module();layer.linear_attn.A_log=torch.nn.Parameter(torch.empty(1,device='meta'))
        layer.norm=torch.nn.Linear(1,1,bias=False,device='meta')
        # A one-dimensional norm parameter, matching model RMSNorm.
        layer.norm.weight=torch.nn.Parameter(torch.empty(1,device='meta'))
        w.load_layer(layer,0)
        self.assertEqual(float(layer.linear_attn.A_log),float(value))
        self.assertEqual(float(layer.norm.weight),0.)
        logits=torch.zeros(2,248320);logits[0,248046]=4.;logits[1,7]=5.
        class Tokenizer:
            def decode(self,ids):return str(ids)
        rows=score_rows(logits,[2,4],[1,2,3,7,8,9],Tokenizer())
        self.assertEqual(rows[0]['expected_next_token'],7)
        self.assertEqual(rows[1]['expected_next_token'],9)
        self.assertEqual(rows[0]['targets'][-1]['rank'],1)

if __name__=='__main__':unittest.main()
