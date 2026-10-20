"""Prefill HC pointwise fusion, preserving explicit bf16 rounding points."""
import torch
import triton
import triton.language as tl


@triton.jit
def _norm(X,W,Y,H:tl.constexpr,HC:tl.constexpr,EPS:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0)
    col=tl.arange(0,B)
    x=tl.load(X+row*H+col,col<H,0).to(tl.float32)
    inv=tl.rsqrt(tl.sum(x*x,0)/H+EPS).to(Y.dtype.element_ty).to(tl.float32)
    xn=(x*inv).to(Y.dtype.element_ty).to(tl.float32)
    gamma=(1+tl.load(W+(row%HC)*H+col,col<H,0).to(tl.float32)).to(Y.dtype.element_ty).to(tl.float32)
    tl.store(Y+row*H+col,xn*gamma,col<H)


@triton.jit
def _collapse(X,G,Y,H:tl.constexpr,HC:tl.constexpr,B:tl.constexpr):
    t=tl.program_id(0)
    col=tl.program_id(1)*B+tl.arange(0,B)
    acc=tl.full((B,),0,tl.float32)
    for stream in range(HC):
        off=(t*HC+stream)*H+col
        x=tl.load(X+off,col<H,0).to(tl.float32)
        g=tl.load(G+off,col<H,0).to(tl.float32)
        gate=tl.sigmoid(g).to(Y.dtype.element_ty).to(tl.float32)
        value=(x*gate).to(Y.dtype.element_ty).to(tl.float32)
        acc+=value
    tl.store(Y+t*H+col,acc/HC,col<H)


@triton.jit
def _combine(R,X,I,Y,H:tl.constexpr,HC:tl.constexpr,B:tl.constexpr):
    t=tl.program_id(0)
    stream=tl.program_id(1)
    col=tl.program_id(2)*B+tl.arange(0,B)
    inj=tl.load(I+t*HC+stream).to(tl.float32)
    scaled=(inj/HC).to(Y.dtype.element_ty).to(tl.float32)
    w=tl.sigmoid(scaled).to(Y.dtype.element_ty).to(tl.float32)*2
    x=tl.load(X+t*H+col,col<H,0).to(tl.float32)
    prod=(x*w).to(Y.dtype.element_ty).to(tl.float32)
    off=(t*HC+stream)*H+col
    residual=tl.load(R+off,col<H,0).to(tl.float32)
    tl.store(Y+off,residual+prod,col<H)


def mix(hc,x):
    t,streams,h=x.shape
    normed=torch.empty_like(x)
    _norm[(t*streams,)](x,hc.hc_norm.weight,normed,h,streams,hc.eps,triton.next_power_of_2(h))
    flat=normed.view(t,streams*h)
    lo=torch.nn.functional.linear(flat,hc.input_mix_weight_down.weight)
    lo=torch.nn.functional.silu(lo/streams)
    gate=torch.nn.functional.linear(lo,hc.input_mix_weight_up.weight)
    out=torch.empty((t,h),device=x.device,dtype=x.dtype)
    _collapse[(t,triton.cdiv(h,256))](flat,gate,out,h,streams,256,enable_fp_fusion=False)
    inject=None
    if hc.has_inject:
        inject=torch.nn.functional.linear(flat,hc.block_inject_weight.weight)
    return out,inject


def combine(hc,residual,block_out,inject):
    t,streams,h=residual.shape
    out=torch.empty_like(residual)
    _combine[(t,streams,triton.cdiv(h,256))](residual,block_out,inject,out,h,streams,256,enable_fp_fusion=False)
    return out
