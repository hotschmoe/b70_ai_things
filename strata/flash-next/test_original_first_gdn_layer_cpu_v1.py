#!/usr/bin/env python3
"""Synthetic asymmetric layer composition properties; no model payload/GPU."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import numpy as np
from original_first_gdn_layer_v1 import FirstGdnLayer,Geometry,source_metadata_contract


class SyntheticRows:
    actual_source=False
    def __init__(self,g):
        self.g=g;self.weights={};self.calls=[];rng=np.random.default_rng(20261010)
        n=g.embd;d=n*g.streams;z=g.state*g.value_heads;c=2*g.state*g.key_heads+z;prefix='blk.0.'
        shapes={'token_embd.weight':(n,13),'attn_qkv.weight':(n,c),'attn_gate.weight':(n,z),
            'ssm_out.weight':(z,n),'ssm_alpha.weight':(n,g.value_heads),'ssm_beta.weight':(n,g.value_heads),
            'ssm_a':(g.value_heads,),'ssm_dt.bias':(g.value_heads,),'ssm_norm.weight':(g.state,),
            'ssm_conv1d.weight':(g.conv_kernel,c),'ffn_gate_inp.weight':(n,g.experts),
            'ffn_gate_inp_shexp.weight':(n,),'ffn_gate_shexp.weight':(n,g.ffn),'ffn_up_shexp.weight':(n,g.ffn),
            'ffn_down_shexp.weight':(g.ffn,n),'ffn_gate_exps.weight':(n,g.ffn,g.experts),
            'ffn_up_exps.weight':(n,g.ffn,g.experts),'ffn_down_exps.weight':(g.ffn,n,g.experts)}
        for half in ('attn','ffn'):
            for role,shape in [('norm',(d,)),('down',(d,g.low_rank)),('up',(g.low_rank,d)),('inject',(d,g.streams))]:shapes['hc_'+half+'_'+role+'.weight']=shape
        for role,shape in shapes.items():
            name=role if role=='token_embd.weight' else prefix+role
            value=rng.normal(0,.15,size=tuple(reversed(shape)))
            if 'norm' in role:value=1+rng.normal(0,.1,size=value.shape)
            if role=='ssm_a':value=-np.exp(rng.normal(0,.1,size=value.shape))
            self.weights[name]=value.astype(np.float32).astype(np.float64)
    def shape(self,name):return tuple(reversed(self.weights[name].shape))
    def rows(self,name,indices):
        data=self.weights[name].reshape(-1,self.shape(name)[0]);result=data[list(indices)].copy();self.calls.append(result.nbytes);return result


def main():
    g=Geometry(embd=6,streams=2,low_rank=3,state=3,key_heads=2,value_heads=4,conv_kernel=4,experts=7,topk=3,ffn=5)
    provider=SyntheticRows(g);model=FirstGdnLayer(provider,g,tile_bytes=128);full=FirstGdnLayer(SyntheticRows(g),g)
    tiled=model.tokens([2,7,3]);untiled=full.tokens([2,7,3]);controls=[]
    for a,b in zip(tiled,untiled):
        assert np.allclose(a['details']['residual'],b['details']['residual'],rtol=0,atol=1e-13)
        assert np.array_equal(a['details']['ffn']['ids'],b['details']['ffn']['ids'])
    assert max(provider.calls)<=128;controls.append('all_decoded_source_tiles_bounded')
    controls.append('row_tiling_matches_untiled_equations')
    # Embeddings are independently gathered and HC streams start as own copies.
    standalone=model.tokens([3])[0]
    assert not np.allclose(standalone['state']['recurrent'],tiled[-1]['state']['recurrent']);controls.append('own_temporal_recurrence_not_reset_each_token')
    repeated=model.tokens([2,7,3]);assert np.array_equal(repeated[-1]['details']['residual'],tiled[-1]['details']['residual']);controls.append('independent_empty_state_replay')
    x=np.arange(g.embd,dtype=float)*.1+.03;out,ffn=model.ffn(x);logits=provider.weights['blk.0.ffn_gate_inp.weight']@x
    expected_ids=np.lexsort((np.arange(g.experts),-logits))[:g.topk];assert np.array_equal(ffn['ids'],expected_ids)
    reconstructed=np.zeros(g.embd)
    for expert,weight in zip(ffn['ids'],ffn['weights']):
        gate=provider.weights['blk.0.ffn_gate_exps.weight'][expert]@x;up=provider.weights['blk.0.ffn_up_exps.weight'][expert]@x
        hidden=gate/(1+np.exp(-gate))*up
        reconstructed+=weight*(provider.weights['blk.0.ffn_down_exps.weight'][expert]@hidden)
    assert np.allclose(reconstructed,ffn['routed'],atol=1e-15);controls.append('own_routing_expert_offsets_weighted_sum')
    assert abs(ffn['weights'].sum()-1)<1e-15;controls.append('selected_expert_softmax_renormalization')
    # Ties must choose lower source expert IDs without geometry/type substitution.
    tie=SyntheticRows(g);tie.weights['blk.0.ffn_gate_inp.weight'].fill(0)
    assert np.array_equal(FirstGdnLayer(tie,g).ffn(x)[1]['ids'],np.arange(g.topk));controls.append('router_tie_lower_id')
    # Compare normalization and HC write to separate, literal scalar expressions.
    residual=np.array([[.1,.2,.3,.4,.5,.6],[-.8,.7,-.6,.5,-.4,.3]])
    hc=model.hc_read(residual,'attn');gamma=provider.weights['blk.0.hc_attn_norm.weight'].reshape(residual.shape)
    expected=np.array([[residual[h,j]*gamma[h,j]/np.sqrt(sum(residual[h]**2)/g.embd+g.eps) for j in range(g.embd)] for h in range(g.streams)])
    assert np.allclose(hc['normalized'],expected,atol=1e-15)
    assert not np.allclose(hc['normalized'],residual*gamma/np.sqrt(np.sum(residual**2,axis=1,keepdims=True)+g.eps));controls.append('hc_stream_mean_not_squared_norm')
    block=np.arange(g.embd)*.01;inject=np.array([.7,-.4]);want=residual+block[None,:]*(2/(1+np.exp(-inject/g.streams)))[:,None]
    assert np.array_equal(model.hc_write(residual,block,inject),want);controls.append('hc_write_injection_divided_by_streams')
    # GDN squared L2, output1/sqrt(S), sigmoid output gate all exposed.
    state=model.initial_state();state['conv']=np.arange(state['conv'].size).reshape(state['conv'].shape)*.001
    state['recurrent']=np.arange(state['recurrent'].size).reshape(state['recurrent'].shape)*.0002
    after,mixed,stage=model.mixer(x,state)
    kh=g.key_heads;ss=g.state;conv=stage['conv_silu'];rawq=conv[:kh*ss].reshape(kh,ss)
    # Official models/models.h:14-17 rms_norm(x,eps/n)/sqrt(n) = x/sqrt(sum(x*x)+eps).
    qwant=rawq/np.sqrt(np.sum(rawq**2,axis=1,keepdims=True)+g.eps);assert np.array_equal(stage['q'],qwant)
    assert not np.allclose(stage['q'],rawq/np.sqrt(np.mean(rawq**2,axis=1,keepdims=True)+g.eps));controls.append('gdn_l2_sum_plus_eps_not_rms')
    for h in range(g.value_heads):
        assert np.allclose(stage['core'][h],(stage['q'][h%kh]@after['recurrent'][h])/np.sqrt(ss),atol=1e-15)
    controls.append('gdn_output_scale_inverse_sqrt_state')
    z=provider.weights['blk.0.attn_gate.weight']@x
    assert np.allclose(stage['gated'],stage['normalized']/(1+np.exp(-z.reshape(g.value_heads,ss))),atol=1e-15);controls.append('gdn_output_sigmoid_not_silu')
    for label,call in [('unsupported_native_lane',lambda:FirstGdnLayer(provider,g,lane='native')),
                       ('wrong_layer',lambda:FirstGdnLayer(provider,g,layer=1)),
                       ('oversized_tile',lambda:FirstGdnLayer(provider,g,tile_bytes=67108865)),
                       ('bad_token_roster',lambda:model.tokens([99]))]:
        try:call()
        except (ValueError,NotImplementedError):controls.append(label)
        else:raise AssertionError('Negative admitted '+label)
    metadata={'qwen4exp.attention.layer_norm_rms_epsilon':{'value':float(np.float32(1e-6))}}
    assert source_metadata_contract(metadata)['effective_expert_scale']==1
    try:source_metadata_contract({**metadata,'qwen4exp.expert_weights_scale':{'value':2}})
    except ValueError:controls.append('nonidentity_expert_scale_rejected')
    else:raise AssertionError('Nonidentity expert scale admitted')
    result={'mode':'CPU_SYNTHETIC_FIRST_GDN_LAYER_EQUATIONS','passed':True,'tokens':3,'controls':controls,
        'max_decoded_tile_bytes':max(provider.calls),'reference_source_sha256':hashlib.sha256(Path(__file__).with_name('original_first_gdn_layer_v1.py').read_bytes()).hexdigest(),
        'operator_lane':'original_fp64','own_embedding_routing_state':True,'actual_model_payloads_read':False,'gpu_executed':False,'native_storage_qualified':False,'full_model_math_qualified':False,'todo_native_contracts':list(model.TODO_NATIVE_CONTRACTS)}
    Path(__file__).with_name('original-first-gdn-layer-cpu-receipt-v1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
