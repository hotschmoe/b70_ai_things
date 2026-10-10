"""Source39 wrong-fingerprint header proposal, never an observed restore proof."""
import struct

MASK=(1<<64)-1
P1=0x9E3779B185EBCA87;P2=0xC2B2AE3D27D4EB4F;P3=0x165667B19E3779F9
P4=0x85EBCA77C2B2AE63;P5=0x27D4EB2F165667C5

def require(ok,message):
 if not ok:raise ValueError(message)

def rol(x,n):
 x &= MASK
 return ((x<<n)|(x>>(64-n)))&MASK

def round1(a,w):return (rol((a+w*P2)&MASK,31)*P1)&MASK

def header_hash56(raw):
 """Exact SessionHasher update/digest for one 56-byte, seed-zero header."""
 require(type(raw)is bytes and len(raw)==56,'Exact header hash extent required')
 lanes=[(P1+P2)&MASK,P2,0,(-P1)&MASK]
 for i in range(4):lanes[i]=round1(lanes[i],struct.unpack_from('<Q',raw,8*i)[0])
 h=sum(rol(l,n) for l,n in zip(lanes,(1,7,12,18)))&MASK
 for l in lanes:h=((h^round1(0,l))*P1+P4)&MASK
 h=(h+56)&MASK
 for i in range(32,56,8):h=(rol(h^round1(0,struct.unpack_from('<Q',raw,i)[0]),27)*P1+P4)&MASK
 h^=h>>33;h=(h*P2)&MASK;h^=h>>29;h=(h*P3)&MASK;h^=h>>32
 return h

def wrong_model_header(original,file_bytes):
 """Only model fingerprint and its checksum change; original file stays intact."""
 require(type(original)is bytes and len(original)==64 and original[:8]==b'STRSESS\x01','Actual saved format-v1 header required')
 version,size,model,config,payload,r0,r1,checksum=struct.unpack_from('<IIQQQQQQ',original,8)
 require(version==1 and size==64 and r0==r1==0 and checksum==header_hash56(original[:56]),'Actual original session header must be valid before negative derivation')
 require(type(file_bytes)is int and file_bytes==64+payload+16 and payload>0,'Actual saved session extent differs')
 changed=bytearray(original);struct.pack_into('<Q',changed,16,model^1);struct.pack_into('<Q',changed,56,header_hash56(bytes(changed[:56])))
 require(bytes(changed[:16])==original[:16] and bytes(changed[24:56])==original[24:56],'Wrong fingerprint changed another header field')
 return bytes(changed),{'original_model_fingerprint':model,'negative_model_fingerprint':model^1,'config_fingerprint':config,'payload_bytes':payload,'only_model_and_header_checksum_changed':True,'native_checksum_implementation_qualified':False,'actual_refusal_observed':False}

def fingerprint_refusal(status,body,parallel):
 require(type(status)is int and type(parallel)is bool and type(body)is dict and set(body)=={'error'},'Actual HTTP refusal shape required')
 error=body['error'];require(type(error)is dict and type(error.get('message'))is str,'Actual structured refusal required')
 if parallel:
  require(status==501 and error['code']==501 and error['type']=='server_error' and error['message']=='slot save/restore is not available with parallel requests ("parallel" / --batch)','Actual enabled parallel persisted branch must refuse exactly')
 else:
  require(status==400 and error['code']==400 and error['type']=='invalid_request_error' and error['message']=='session file: saved with another model (model fingerprint differs)' and error.get('kind')=='invalid' and 'published' not in error,'Wrong-model restore must reach native fingerprint refusal, not checksum/path/disabled branch')
 return {'actual_HTTP_refusal_shape_matched':True,'current_source_recipe_and_owned_execution_still_required':True,'cached_state_or_math_qualified':False,'full_cache_runtime_qualified':False}
