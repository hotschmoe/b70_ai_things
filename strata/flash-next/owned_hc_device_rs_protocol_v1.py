"""Own argument/result protocol; no captured operands or correction tables."""
import hashlib,re,struct
from pathlib import Path
import numpy as np
def require(ok,msg):
 if not ok:raise ValueError(msg)
def arguments(values):
 a=np.asarray(values,dtype='<f4');require(a.shape==(4,) and np.isfinite(a).all() and (a>0).all(),'Four finite positive owned F32 arguments required');return a
def words(raw):return [f'{v:08x}' for v in struct.unpack('<4I',raw)]
def request(sequence,values):
 require(type(sequence) is int and 1<=sequence<=512,'Exact bounded next own ordinal required');a=arguments(values);return 'RSQRT '+str(sequence)+' '+' '.join(words(a.tobytes()))+'\n'
def response(line,sequence,values):
 match=re.fullmatch(r'OWNRS37_RESPONSE seq=([1-9][0-9]*)(?: arg=([0-9a-f]{8})){4}(?: rs=([0-9a-f]{8})){4} direct_replays_bitwise=1\n',line);require(match is not None,'Exact complete own RS response required');tokens=line.split();require(tokens[1]=='seq='+str(sequence),'Foreign/stale own ordinal');arg=[v[4:] for v in tokens[2:6]];require(arg==words(arguments(values).tobytes()),'Actual helper echoed a foreign or substituted argument');rs=[int(v[3:],16) for v in tokens[6:10]];result=np.frombuffer(struct.pack('<4I',*rs),dtype='<f4').copy();arguments(result);return result
class OwnDeviceRsClient:
 def __init__(self,exchange,qualification):
  require(callable(exchange) and qualification['helper_source_sha256']==hashlib.sha256(Path(__file__).with_name('owned_hc_rsqrt37_service_v1.cpp').read_bytes()).hexdigest(),'Current exact own service source required');require(qualification['operation']=='sycl::rsqrt' and qualification['captured_operands_used'] is False and qualification['ULP_adjustment_or_lookup_used'] is False,'Own arithmetic reference scope differs');self.exchange=exchange;self.qualification=dict(qualification);self.records=[]
 def require_actual(self):require(self.qualification.get('actual_owned_device_started') is True and self.qualification.get('synthetic_only') is False and self.qualification.get('actual_source_flags_and_recipe_qualified') is True,'Actual original model requires real owned device/flags admission')
 def rsqrt_owned(self,values,role):
  require(re.fullmatch(r'blk\.(?:[0-9]|[1-3][0-9]|4[0-7])\.hc_(?:attn|ffn)_|output_hc_',role) is not None,'Exact original HC tensor role required');a=arguments(values);sequence=len(self.records)+1;command=request(sequence,a);line=self.exchange(command);rs=response(line,sequence,a);self.records.append({'sequence':sequence,'role':role,'argument_LE_F32_hex':a.tobytes().hex(),'result_LE_F32_hex':rs.tobytes().hex(),'request':command,'response':line,'argument_sha256':hashlib.sha256(a.tobytes()).hexdigest(),'result_sha256':hashlib.sha256(rs.tobytes()).hexdigest(),'captured_operands_used':False});return rs
