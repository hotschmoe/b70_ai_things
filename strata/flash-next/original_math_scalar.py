"""Independent scalar equation building blocks, not an end-to-end model oracle.

original_fp64 evaluates exact source values in FP64. declared_storage rounds only
named storage boundaries; GPU reduction ordering/native transcendental accuracy
is NOT emulated and must not be claimed byte exact from these functions.
"""
import numpy as np


def sigmoid(x):
    x=np.asarray(x,dtype=np.float64);return np.exp(-np.logaddexp(0,-x))


def store(x,lane,format='F32'):
    if lane=='original_fp64':return np.asarray(x,dtype=np.float64)
    if lane!='declared_storage':raise ValueError('Unknown arithmetic lane')
    x=np.asarray(x,dtype=np.float64)
    if format=='F16':return x.astype(np.float16).astype(np.float64)
    if format=='F32':return x.astype(np.float32).astype(np.float64)
    raise ValueError('Undeclared storage format')


def gdn_step(state,q,k,v,log_decay,beta,lane='original_fp64'):
    """Source layout independent: state[head,i,j], modulo value-to-key pairing."""
    state=np.array(state,dtype=np.float64,copy=True);q=np.asarray(q,dtype=np.float64);k=np.asarray(k,dtype=np.float64);v=np.asarray(v,dtype=np.float64)
    heads,size,_=state.shape
    if q.shape!=k.shape or q.shape[1]!=size or v.shape!=(heads,size) or state.shape!=(heads,size,size) or heads%len(q):raise ValueError('GDN shape mismatch')
    output=np.empty_like(v)
    for h in range(heads):
        key=h%len(k);decayed=state[h]*np.exp(log_decay[h])
        delta=(v[h]-k[key]@decayed)*beta[h]
        state[h]=store(decayed+np.outer(k[key],delta),lane)
        output[h]=store(q[key]@state[h],lane)
    return state,output


def labelled_conv(history,current,weights,lane='original_fp64'):
    """Weights oldest-first; current input is newest kernel row."""
    history=np.asarray(history,dtype=np.float64);current=np.asarray(current,dtype=np.float64);weights=np.asarray(weights,dtype=np.float64)
    if history.shape!=(len(weights)-1,len(current)) or weights.shape[1:]!=current.shape:raise ValueError('Convolution history geometry mismatch')
    full=np.concatenate([history,current[None]],axis=0)
    output=store(np.sum(full*weights,axis=0),lane)
    return store(full[1:],lane),output


def ple_dilated_conv(history,current,weights,lane='original_fp64'):
    """Actual 4-row kernel with dilation3 and 9 retained normalized rows."""
    history=np.asarray(history,dtype=np.float64);current=np.asarray(current,dtype=np.float64);weights=np.asarray(weights,dtype=np.float64)
    if history.shape!=(9,len(current)) or weights.shape!=(4,len(current)):raise ValueError('PLE history/kernel geometry mismatch')
    selected=np.stack([history[0],history[3],history[6],current])
    return store(np.concatenate([history[1:],current[None]]),lane),store(np.sum(selected*weights,axis=0),lane)


def qsa_indexer_scores(pooled_keys,queries,bias):
    """ReLU per query head, then head sum plus block bias."""
    pooled_keys=np.asarray(pooled_keys,dtype=np.float64);queries=np.asarray(queries,dtype=np.float64)
    return np.maximum(queries@pooled_keys.T,0).sum(axis=0)+np.asarray(bias)


def qsa_selected_attention(query,keys,values,selected,gate,lane='original_fp64'):
    """Selected-cell causal contract; caller must provide independently selected IDs.

    declared_storage reads F16 KV values, then materializes gated output F16.
    Indexer/position/pooling selection is not inferred by this local building block.
    """
    query=np.asarray(query,dtype=np.float64);keys=store(keys,lane,'F16');values=store(values,lane,'F16');gate=np.asarray(gate,dtype=np.float64)
    nh,size=query.shape;cells,nkv,dim=keys.shape
    if dim!=size or values.shape!=keys.shape or nh%nkv or gate.shape!=query.shape:raise ValueError('QSA shape mismatch')
    if len(set(selected))!=len(selected) or not selected or any(type(i) is not int or not 0<=i<cells for i in selected):raise ValueError('Invalid selected-cell roster')
    out=np.empty_like(query)
    for h in range(nh):
        kh=h//(nh//nkv);logits=keys[selected,kh]@query[h]/np.sqrt(size);prob=np.exp(logits-logits.max());prob/=prob.sum()
        out[h]=store((prob@values[selected,kh])*sigmoid(gate[h]),lane,'F16')
    return out


def metrics(got,reference):
    got=np.asarray(got,dtype=np.float64);reference=np.asarray(reference,dtype=np.float64)
    if got.shape!=reference.shape or not np.isfinite(got).all() or not np.isfinite(reference).all():raise ValueError('Nonfinite/shape comparison')
    err=got-reference
    return {'nmse':float(np.sum(err*err)/max(1e-30,np.sum(reference*reference))),
            'max_normalized':float(np.max(np.abs(err),initial=0)/max(1e-6,np.max(np.abs(reference),initial=0)))}


def rms_weighted(rows,weights,eps=float(np.float32(1e-6)),lane='original_fp64'):
    rows=np.asarray(rows,dtype=np.float64);weights=np.asarray(weights,dtype=np.float64)
    if rows.shape!=weights.shape:raise ValueError('RMS source shape differs')
    return store(rows*weights/np.sqrt(np.mean(rows*rows,axis=-1,keepdims=True)+eps),lane)


def ple_postprojection(key,value,residual,norm_key,norm_query,norm_conv,history,conv_weights,lane='original_fp64'):
    """Independent PLE equations after exact-original-weight key/value projections.

    Logical history shape [9,10240]; physical source storage is row-fastest
    [channel,9] and must be explicitly transposed by the caller.
    """
    residual=np.asarray(residual,dtype=np.float64);key=np.asarray(key,dtype=np.float64);value=np.asarray(value,dtype=np.float64)
    if residual.shape!=(4,2560) or key.shape!=residual.shape or value.shape!=(2560,):raise ValueError('Actual PLE projection geometry differs')
    kn=rms_weighted(key,norm_key,lane=lane);qn=rms_weighted(residual,norm_query,lane=lane)
    score=np.sum(kn*qn,axis=1)/np.sqrt(2560)
    gate=store(sigmoid(np.sign(score)*np.sqrt(np.maximum(np.abs(score),1e-6))),lane)
    gated=store(gate[:,None]*value,lane);normalized=rms_weighted(gated,norm_conv,lane=lane)
    next_history,convolved=ple_dilated_conv(history,normalized.reshape(-1),conv_weights,lane)
    convolved=convolved.reshape(4,2560)
    result=store(residual+gated+convolved*sigmoid(convolved),lane)
    return {'key_norm':kn,'query_norm':qn,'gate':gate,'gated':gated,'normalized':normalized,'convolved':convolved,'result':result,'next_history':next_history}
