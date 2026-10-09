#!/usr/bin/env python3
"""Equation property controls only; not actual model/kernel numerical results."""
import json
import hashlib
from pathlib import Path
import numpy as np
from original_math_scalar import gdn_step,labelled_conv,ple_dilated_conv,qsa_indexer_scores,qsa_selected_attention,metrics


def main():
    rng=np.random.default_rng(17);checks=[]
    state=rng.normal(size=(4,3,3));q=rng.normal(size=(2,3));k=rng.normal(size=(2,3));v=rng.normal(size=(4,3));decay=np.array([-.2,-.4,-.6,-.8]);beta=np.array([.2,.4,.6,.8])
    correct,out=gdn_step(state,q,k,v,decay,beta)
    rival=state.copy()
    for h in range(4):
        i=h//2;st=state[h]*np.exp(decay[h]);rival[h]=st+np.outer(k[i],(v[h]-k[i]@st)*beta[h])
    assert not np.allclose(correct,rival);checks.append('gdn_modulo_pairing')
    late=state.copy()
    for h in range(4):
        i=h%2;late[h]=(state[h]+np.outer(k[i],(v[h]-k[i]@state[h])*beta[h]))*np.exp(decay[h])
    assert not np.allclose(correct,late);checks.append('gdn_decay_before_update')
    history=np.arange(6,dtype=float).reshape(3,2);current=np.array([9,11]);weights=np.array([[1,2],[3,4],[5,6],[7,8]])
    shifted,conv=labelled_conv(history,current,weights);assert np.array_equal(shifted[-1],current) and not np.array_equal(conv,(np.concatenate([history,current[None]])*weights[::-1]).sum(0));checks.append('conv_oldest_first')
    h=np.arange(18,dtype=float).reshape(9,2);shifted,c=ple_dilated_conv(h,current,weights);assert np.array_equal(c,(np.stack([h[0],h[3],h[6],current])*weights).sum(0));assert len(shifted)==9;checks.append('ple_9_rows_dilation3')
    scores=qsa_indexer_scores(np.array([[1.,-2.],[-3.,4.]]),np.array([[1.,0.],[-1.,0.]]),np.array([0.,0.]));wrong=np.maximum((np.array([[1.,0.],[-1.,0.]])@np.array([[1.,-2.],[-3.,4.]]).T).sum(0),0);assert not np.array_equal(scores,wrong);checks.append('qsa_relu_per_head')
    query=np.ones((4,2));keys=rng.normal(size=(3,2,2));values=rng.normal(size=(3,2,2));gate=np.zeros_like(query)
    a=qsa_selected_attention(query,keys,values,[0,2],gate);b=qsa_selected_attention(query,keys,values,[0,1,2],gate);assert not np.array_equal(a,b);checks.append('qsa_selected_only')
    mapped=qsa_selected_attention(np.zeros((24,256)),np.zeros((1,2,256)),np.stack([np.ones(256),np.ones(256)*7])[None],[0],np.zeros((24,256)))
    expected=np.concatenate([np.ones((12,256))*0.5,np.ones((12,256))*3.5])
    modulo_rival=np.stack([np.ones(256)*(0.5 if h%2==0 else 3.5) for h in range(24)])
    assert np.array_equal(mapped,expected) and not np.array_equal(mapped,modulo_rival);checks.append('qsa_division_pairing_modulo_rival')
    native=qsa_selected_attention(query,keys,values,[0,2],gate,'declared_storage');cost=metrics(native,a);assert cost['nmse']>0;checks.append('separate_storage_difference')
    try:qsa_selected_attention(query,keys,values,[0,0],gate)
    except ValueError:checks.append('duplicate_selected_rejection')
    else:raise AssertionError('Duplicate selected admitted')
    result={'mode':'CPU_SCALAR_EQUATION_PROPERTIES_ONLY','passed':True,'controls':checks,'qsa_declared_storage_vs_original_math':cost,'gpu_executed':False,'full_model_math_qualified':False,'gpu_native_arithmetic_emulated':False,'reference_source_sha256':hashlib.sha256(Path(__file__).with_name('original_math_scalar.py').read_bytes()).hexdigest()}
    Path(__file__).with_name('original-math-scalar-cpu-receipt-v1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
