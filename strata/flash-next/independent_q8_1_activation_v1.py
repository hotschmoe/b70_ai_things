"""Independent CPU native / official-GGML Q8_1 packet contracts; no SYCL imports.

Input is explicitly LE F32 raw bytes, row-major, contiguous block32. Packet is
LE half d, LE half s,32 signed codes;36 bytes. Packet encoder is independent of
SwiGLU/native-exp reconstruction and must never claim that hidden was observed.
"""
import numpy as np

NATIVE='native_raw_f32_xor_sum_division'
OFFICIAL='official_ggml_integer_code_sum_reciprocal'
QFUSE='gdn_qfuse_unclamped_raw_f32_sum_division'


def encode(raw,rows,cols,contract=NATIVE):
    if type(rows) is not int or type(cols) is not int or rows<=0 or cols<=0 or cols%32 or len(raw)!=rows*cols*4 or len(raw)>64*1024*1024:raise ValueError('F32 row/block shape or bounded input extent differs')
    if contract not in (NATIVE,OFFICIAL,QFUSE):raise ValueError('Unknown activation contract')
    x=np.frombuffer(raw,dtype='<f4').reshape(-1,32)
    if not np.isfinite(x).all():raise ValueError('Nonfinite input marker; no finite packet qualification')
    amax=np.max(np.abs(x),axis=1);d=np.asarray(amax/np.float32(127),dtype=np.float32)
    if np.any((amax!=0)&(d==0)):raise ValueError('Nonzero scale underflow; source division/round cast is not a finite reference contract')
    if contract==NATIVE:d=np.minimum(d,np.float32(65504))
    divisor=np.where(amax==0,np.float32(1),d)
    with np.errstate(over='ignore',invalid='ignore'):
        if contract!=OFFICIAL:quotient=x/divisor[:,None]
        else:quotient=x*np.asarray(np.float32(1)/divisor,dtype=np.float32)[:,None]
        if not np.isfinite(quotient).all():raise ValueError('Nonfinite quotient is not an independently qualified finite cast contract')
        rounded=np.sign(quotient)*np.floor(np.abs(quotient.astype(np.float64))+.5)
        codes=np.clip(rounded,-127,127).astype(np.int8)
        if contract!=OFFICIAL:
            lanes=x.copy();positions=np.arange(32)
            for mask in (16,8,4,2,1):lanes=np.asarray(lanes+lanes[:,positions^mask],dtype=np.float32)
            total=lanes[:,0]
            if np.isnan(total).any():raise ValueError('Finite inputs produced NaN reduction; preserve marker, not finite qualification')
            if contract==NATIVE:total=np.clip(total,-65504,65504)
        else:total=np.asarray(np.sum(codes.astype(np.int32),axis=1)*d,dtype=np.float32)
        half_d=d.astype('<f2');half_s=total.astype('<f2')
    packet=np.empty((len(x),36),dtype=np.uint8);packet[:,:2]=half_d.view(np.uint8).reshape(-1,2);packet[:,2:4]=half_s.view(np.uint8).reshape(-1,2);packet[:,4:]=codes.view(np.uint8)
    return packet.tobytes()


def decode(packet,rows,cols):
    if type(rows) is not int or type(cols) is not int or rows<=0 or cols<=0 or cols%32 or len(packet)!=rows*(cols//32)*36:raise ValueError('Q8_1 packet shape differs')
    blocks=np.frombuffer(packet,dtype=np.uint8).reshape(-1,36);d=blocks[:,:2].copy().view('<f2').reshape(-1,1).astype(np.float64);s=blocks[:,2:4].copy().view('<f2').reshape(-1).astype(np.float64)
    reconstructed=(d*blocks[:,4:].view(np.int8)).reshape(rows,cols)
    return {'reconstructed_activation':reconstructed,'scale':d[:,0],'stored_sum':s,'codes':blocks[:,4:].view(np.int8).copy()}


def quantization_cost(raw,packet,rows,cols):
    x=np.frombuffer(raw,dtype='<f4').astype(np.float64).reshape(rows,cols);decoded=decode(packet,rows,cols);q=decoded['reconstructed_activation']
    if not np.isfinite(q).all() or not np.isfinite(decoded['stored_sum']).all():raise ValueError('Nonfinite packet cannot qualify finite cost')
    difference=q-x
    return {'nmse':float(np.sum(difference*difference)/max(1e-30,np.sum(x*x))),
        'max_normalized':float(np.max(np.abs(difference))/max(1e-6,np.max(np.abs(x)))),
        'scope':'activation reconstruction only; stored-sum-dependent vec-dot corrections excluded',
        'implementation_error_measured':False}


def require_packet_exact(raw,packet,rows,cols):
    expected=encode(raw,rows,cols,NATIVE)
    if packet!=expected:raise ValueError('Native Q8_1 packet differs from independently encoded supplied F32 source')
    return {'packet_exact':True,'raw_fused_hidden_observed':False,'gpu_math_qualified':False}
