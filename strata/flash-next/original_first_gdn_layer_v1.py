"""Independent original-GGUF first GDN layer equations, bounded CPU reference.

No Strata arithmetic. Only original_fp64 is supported. Native Q8_1/F32 storage
and production reduction/intrinsic contracts are explicitly unimplemented.
"""
from dataclasses import dataclass
import math
import numpy as np
from original_gguf_reference import OriginalGguf
from original_gguf_vector_decoder_v2 import rows as vector_rows


@dataclass(frozen=True)
class Geometry:
    embd:int=2560
    streams:int=4
    low_rank:int=320
    state:int=128
    key_heads:int=16
    value_heads:int=48
    conv_kernel:int=4
    experts:int=512
    topk:int=10
    ffn:int=640
    eps:float=float(np.float32(1e-6))


def source_metadata_contract(metadata):
    # llama-hparams stores zero sentinel; graph skips scale when zero or one.
    scale=metadata.get('qwen4exp.expert_weights_scale',{'value':0.0})['value']
    eps=metadata['qwen4exp.attention.layer_norm_rms_epsilon']['value']
    if type(scale) not in (int,float) or scale not in (0,1):raise ValueError('Nonidentity expert scale not implemented')
    if type(eps) not in (int,float) or eps!=float(np.float32(1e-6)):raise ValueError('Original RMS epsilon differs')
    return {'expert_scale_parameter':float(scale),'effective_expert_scale':1.0,'original_rms_epsilon':eps}


class OriginalTensorRows:
    """Actual source-bound provider; construction/read admission uses frozen v1.

    No actual payload is read until rows/project is called. Do not invoke during
    parent GPU controls; current identity and sentinel guards remain caller-owned.
    """
    actual_source=True
    def __init__(self,manifest,identity):
        self.reader=OriginalGguf(manifest,identity)
        self.source_contract=source_metadata_contract(self.reader.files[0]['metadata'])
    def shape(self,name):return tuple(self.reader.tensors[name][1]['shape_ggml_order'])
    def rows(self,name,indices):return vector_rows(self.reader,name,list(indices),'original_fp64')


class FirstGdnLayer:
    TODO_NATIVE_CONTRACTS=('ordinary Q8_1 activation encoder','shared/expert hidden Q8_1 encoder',
        'native F32 reductions/FMA/native exp','prefill F16/BF16 precision seam','production route rounding/ties')
    def __init__(self,provider,geometry=Geometry(),layer=0,tile_bytes=64*1024*1024,lane='original_fp64'):
        if lane!='original_fp64':raise NotImplementedError('Only original_fp64 equations are implemented; native contract is unqualified')
        if layer!=0:raise ValueError('This composition owns first GDN layer0 only')
        if type(tile_bytes) is not int or not 0<tile_bytes<=64*1024*1024:raise ValueError('Decoded tile bound invalid')
        g=geometry
        if g.streams<2 or g.state<1 or g.key_heads<1 or g.value_heads%g.key_heads or not 0<g.topk<=g.experts or g.conv_kernel<2:raise ValueError('Invalid first GDN geometry')
        if provider.actual_source and g!=Geometry():raise ValueError('Actual qwen4exp geometry differs')
        self.p=provider;self.g=g;self.tile_bytes=tile_bytes;self.lane=lane;self.prefix='blk.0.'
        self.validate_roles()

    def validate_roles(self):
        g=self.g;n=g.embd;d=n*g.streams;z=g.value_heads*g.state;c=2*g.key_heads*g.state+z
        expected={'token_embd.weight':(n,None),self.prefix+'attn_qkv.weight':(n,c),self.prefix+'attn_gate.weight':(n,z),
            self.prefix+'ssm_out.weight':(z,n),self.prefix+'ssm_alpha.weight':(n,g.value_heads),
            self.prefix+'ssm_beta.weight':(n,g.value_heads),self.prefix+'ssm_a':(g.value_heads,),
            self.prefix+'ssm_dt.bias':(g.value_heads,),self.prefix+'ssm_norm.weight':(g.state,),
            self.prefix+'ssm_conv1d.weight':(g.conv_kernel,c),self.prefix+'ffn_gate_inp.weight':(n,g.experts),
            self.prefix+'ffn_gate_inp_shexp.weight':(n,),self.prefix+'ffn_gate_shexp.weight':(n,g.ffn),
            self.prefix+'ffn_up_shexp.weight':(n,g.ffn),self.prefix+'ffn_down_shexp.weight':(g.ffn,n),
            self.prefix+'ffn_gate_exps.weight':(n,g.ffn,g.experts),self.prefix+'ffn_up_exps.weight':(n,g.ffn,g.experts),
            self.prefix+'ffn_down_exps.weight':(g.ffn,n,g.experts)}
        for half in ('attn','ffn'):
            for role,shape in [('norm',(d,)),('down',(d,g.low_rank)),('up',(g.low_rank,d)),('inject',(d,g.streams))]:expected[self.prefix+'hc_'+half+'_'+role+'.weight']=shape
        for name,want in expected.items():
            got=self.p.shape(name)
            if len(got)!=len(want) or any(w is not None and w!=a for a,w in zip(got,want)):raise ValueError('Source role/shape differs: '+name)

    def vector(self,role):
        shape=self.p.shape(role)
        if len(shape)!=1 or shape[0]*8>self.tile_bytes:raise ValueError('Vector read exceeds bounded role')
        return self.p.rows(role,[0])[0]

    def project(self,name,x,expert=None):
        x=np.asarray(x,dtype=np.float64);shape=self.p.shape(name)
        if x.ndim!=1 or not np.isfinite(x).all() or len(shape) not in (2,3) or len(x)!=shape[0]:raise ValueError('Projection source/input differs')
        if (len(shape)==3)!=(expert is not None):raise ValueError('Expert role/index missing or unexpected')
        if expert is not None and (type(expert) is not int or not 0<=expert<shape[2]):raise ValueError('Expert index outside source')
        rows=shape[1];tile=self.tile_bytes//(shape[0]*8)
        if tile<1:raise ValueError('One decoded source row exceeds tile cap')
        result=np.empty(rows,dtype=np.float64);offset=(expert*rows if expert is not None else 0)
        for start in range(0,rows,tile):
            count=min(tile,rows-start);weights=self.p.rows(name,range(offset+start,offset+start+count))
            if weights.shape!=(count,shape[0]) or not np.isfinite(weights).all():raise ValueError('Decoded projection tile differs')
            result[start:start+count]=weights@x
        return result

    @staticmethod
    def sigmoid(x):return np.exp(-np.logaddexp(0,-np.asarray(x,dtype=np.float64)))
    @classmethod
    def silu(cls,x):return np.asarray(x,dtype=np.float64)*cls.sigmoid(x)

    def hc_read(self,residual,half):
        g=self.g;r=np.asarray(residual,dtype=np.float64)
        if r.shape!=(g.streams,g.embd) or not np.isfinite(r).all():raise ValueError('HC residual geometry differs')
        stem=self.prefix+'hc_'+half+'_';norm=self.vector(stem+'norm.weight').reshape(r.shape)
        xn=r*norm/np.sqrt(np.mean(r*r,axis=1,keepdims=True)+g.eps)
        down=self.project(stem+'down.weight',xn.reshape(-1));lo=self.silu(down/g.streams)
        gate=self.project(stem+'up.weight',lo).reshape(r.shape)
        inject=self.project(stem+'inject.weight',xn.reshape(-1))
        mixed=np.mean(xn*self.sigmoid(gate),axis=0)
        return {'normalized':xn,'down':down,'low_silu':lo,'gate':gate,'inject':inject,'mixed':mixed}

    def hc_write(self,residual,block,inject):
        g=self.g
        if np.shape(block)!=(g.embd,) or np.shape(inject)!=(g.streams,):raise ValueError('HC write dimensions differ')
        return residual+block[None,:]*(2*self.sigmoid(inject/g.streams))[:,None]

    def initial_state(self):
        g=self.g;c=2*g.key_heads*g.state+g.value_heads*g.state
        return {'recurrent':np.zeros((g.value_heads,g.state,g.state)), 'conv':np.zeros((g.conv_kernel-1,c))}

    def mixer(self,x,state):
        g=self.g;s=g.state;hk=g.key_heads;hv=g.value_heads;c=2*hk*s+hv*s
        recurrent=np.asarray(state['recurrent'],dtype=np.float64);history=np.asarray(state['conv'],dtype=np.float64)
        if recurrent.shape!=(hv,s,s) or history.shape!=(g.conv_kernel-1,c) or not np.isfinite(recurrent).all() or not np.isfinite(history).all():raise ValueError('Owned GDN state dimensions/values differ')
        qkv=self.project(self.prefix+'attn_qkv.weight',x);z=self.project(self.prefix+'attn_gate.weight',x).reshape(hv,s)
        alpha=self.project(self.prefix+'ssm_alpha.weight',x);beta=self.sigmoid(self.project(self.prefix+'ssm_beta.weight',x))
        log_decay=np.logaddexp(0,alpha+self.vector(self.prefix+'ssm_dt.bias'))*self.vector(self.prefix+'ssm_a')
        # GGUF ne0 is kernel tap; one channel per row, oldest tap first.
        full=np.concatenate([history,qkv[None,:]]);raw_conv=np.empty(c)
        conv_tile=self.tile_bytes//(g.conv_kernel*8)
        if conv_tile<1:raise ValueError('One decoded convolution channel exceeds cap')
        for start in range(0,c,conv_tile):
            count=min(conv_tile,c-start)
            weights=self.p.rows(self.prefix+'ssm_conv1d.weight',range(start,start+count)).T
            raw_conv[start:start+count]=np.sum(full[:,start:start+count]*weights,axis=0)
        convolved=self.silu(raw_conv)
        q=convolved[:hk*s].reshape(hk,s);k=convolved[hk*s:2*hk*s].reshape(hk,s);v=convolved[2*hk*s:].reshape(hv,s)
        q=q/np.sqrt(np.sum(q*q,axis=1,keepdims=True)+g.eps);k=k/np.sqrt(np.sum(k*k,axis=1,keepdims=True)+g.eps)
        next_state=recurrent.copy();core=np.empty_like(v)
        for h in range(hv):
            kh=h%hk;decayed=recurrent[h]*np.exp(log_decay[h]);delta=(v[h]-k[kh]@decayed)*beta[h]
            next_state[h]=decayed+np.outer(k[kh],delta);core[h]=(q[kh]@next_state[h])/math.sqrt(s)
        gamma=self.vector(self.prefix+'ssm_norm.weight')
        normalized=core*gamma/np.sqrt(np.mean(core*core,axis=1,keepdims=True)+g.eps)
        gated=normalized*self.sigmoid(z);output=self.project(self.prefix+'ssm_out.weight',gated.reshape(-1))
        return {'recurrent':next_state,'conv':full[1:].copy()},output,{'qkv':qkv,'conv_silu':convolved,'q':q,'k':k,'v':v,'log_decay':log_decay,'beta':beta,'core':core,'normalized':normalized,'gated':gated,'output':output}

    def expert(self,x,expert,shared=False):
        suffix='_shexp.weight' if shared else '_exps.weight';kw={} if shared else {'expert':int(expert)}
        gate=self.project(self.prefix+'ffn_gate'+suffix,x,**kw);up=self.project(self.prefix+'ffn_up'+suffix,x,**kw)
        hidden=self.silu(gate)*up;out=self.project(self.prefix+'ffn_down'+suffix,hidden,**kw)
        return out,{'gate':gate,'up':up,'hidden':hidden,'output':out}

    def ffn(self,x):
        g=self.g;logits=self.project(self.prefix+'ffn_gate_inp.weight',x)
        # Source mathematical softmax/top-k, selected renormalization; lower-ID tie.
        ids=np.lexsort((np.arange(g.experts),-logits))[:g.topk]
        probabilities=np.exp(logits-logits.max());probabilities/=probabilities.sum()
        weights=probabilities[ids]/probabilities[ids].sum()
        routed=np.zeros(g.embd);expert_outputs={}
        for i,w in zip(ids,weights):
            out,detail=self.expert(x,int(i));routed+=w*out;expert_outputs[int(i)]=detail
        shared,shared_detail=self.expert(x,None,shared=True)
        shared_gate=float(self.sigmoid(self.vector(self.prefix+'ffn_gate_inp_shexp.weight')@x))
        output=routed+shared_gate*shared
        return output,{'router_logits':logits,'router_probabilities':probabilities,'ids':ids,'weights':weights,'experts':expert_outputs,'routed':routed,'shared':shared_detail,'shared_gate':shared_gate,'output':output}

    def step(self,residual,state):
        attn=self.hc_read(residual,'attn');next_state,block,mixer=self.mixer(attn['mixed'],state)
        after_attn=self.hc_write(residual,block,attn['inject']);ffn_read=self.hc_read(after_attn,'ffn')
        out,ffn=self.ffn(ffn_read['mixed']);result=self.hc_write(after_attn,out,ffn_read['inject'])
        if not np.isfinite(result).all():raise ValueError('Nonfinite first-layer result')
        return result,next_state,{'attn_hc':attn,'mixer':mixer,'after_attention':after_attn,'ffn_hc':ffn_read,'ffn':ffn,'residual':result}

    def tokens(self,token_ids):
        """Own embedding, routing and sequential state from empty state; layer0 ONLY."""
        ids=list(token_ids)
        if not 1<=len(ids)<=8 or any(type(t) is not int or not 0<=t<self.p.shape('token_embd.weight')[1] for t in ids):raise ValueError('Bounded token roster differs')
        state=self.initial_state();rows=[]
        for token in ids:
            embedding=self.p.rows('token_embd.weight',[token])[0]
            residual=np.broadcast_to(embedding,(self.g.streams,self.g.embd)).copy()
            result,state,details=self.step(residual,state);rows.append({'token':token,'details':details,'state':{k:v.copy() for k,v in state.items()}})
        return rows
