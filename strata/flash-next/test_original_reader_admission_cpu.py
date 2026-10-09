"""Synthetic read-admission failures; no actual model/header/payload reads.

The fake inventory exercises admission/read independently of qwen4exp role
validation, which has its own complete actual-inventory negative controls.
"""
import copy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
from unittest.mock import patch
from original_gguf_reference import OriginalGguf


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run():
    rejected=[]
    with tempfile.TemporaryDirectory(prefix='gguf-admission-') as temporary:
        root=Path(temporary);bank=root/'UD-Q4_K_XL';bank.mkdir();files=[];rows=[];lockfiles=[]
        for i in range(4):
            p=bank/('synthetic-%05d-of-00004.gguf'%(i+1));header=b'SYNTHETIC-HEADER';raw=(struct.pack('<e',0.25)+bytes([1]*32))*16;p.write_bytes(header+raw);st=p.stat()
            files.append({'path':str(p),'header_bytes_read':len(header),'header_sha256':hashlib.sha256(header).hexdigest(),'tensor_data_offset':len(header),'metadata':{},'tensors':[{'name':'synthetic.%d'%i,'type':'Q8_0','shape_ggml_order':[256,2],'elements':512,'packed_bytes':544,'absolute_offset':len(header)}]})
            sig=[st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns]
            rows.append({'path':str(p),'passed':True,'sha256':sha(p),'expected_sha256':sha(p),'stat_before':sig,'stat_after':sig})
            lockfiles.append({'path':'UD-Q4_K_XL/'+p.name,'size':st.st_size,'sha256':sha(p)})
        lock=root/'lock.json';lock.write_text(json.dumps({'revision':'synthetic','files':lockfiles}))
        inventory=root/'inventory.json';manifest=root/'manifest.json';identity=root/'identity.json'
        base={'files':files};receipt={'passed':True,'model_revision':'synthetic','lock_sha256':sha(lock),'rows':rows}
        def write(inv=base,rec=receipt):
            inventory.write_text(json.dumps(inv));identity.write_text(json.dumps(rec));manifest.write_text(json.dumps({'inventory':str(inventory),'inventory_sha256':sha(inventory),'lock':str(lock),'lock_sha256':sha(lock),'max_decoded_or_read_bytes_per_call':4096}))
        def reject(label,fn):
            try:fn()
            except (ValueError,KeyError):rejected.append(label)
            else:raise AssertionError('Reader negative admitted '+label)
        with patch.object(OriginalGguf,'validate_roles',return_value=None):
            write();reader=OriginalGguf(manifest,identity);assert reader.rows('synthetic.0',[0]).shape==(1,256)
            reject('caller_cap_override',lambda:OriginalGguf(manifest,identity,8192))
            reject('caller_cap_nonpositive',lambda:OriginalGguf(manifest,identity,0))
            tiny=OriginalGguf(manifest,identity,512)
            reject('decoded_extent_exceeds_cap_packed_fits',lambda:tiny.rows('synthetic.0',[0]))
            reject('row_outside_extent',lambda:reader.rows('synthetic.0',[2]))
            for label,mutate in [('missing_receipt_row',lambda r:r['rows'].pop()),('missing_receipt_pass',lambda r:r.pop('passed')),('bad_publisher_sha',lambda r:r['rows'][0].__setitem__('sha256','0'*64)),('changed_stat',lambda r:r['rows'][0]['stat_after'].__setitem__(4,0)),('wrong_lock_binding',lambda r:r.__setitem__('lock_sha256','0'*64))]:
                bad=copy.deepcopy(receipt);mutate(bad);write(rec=bad);reject(label,lambda:OriginalGguf(manifest,identity))
            for label,mutate in [('header_hash_mismatch',lambda x:x['files'][0].__setitem__('header_sha256','0'*64)),('tensor_offset_outside_file',lambda x:x['files'][0]['tensors'][0].__setitem__('absolute_offset',9999)),('unknown_source_format',lambda x:x['files'][0]['tensors'][0].__setitem__('type','Q6_K'))]:
                bad=copy.deepcopy(base);mutate(bad);write(inv=bad);reject(label,lambda:OriginalGguf(manifest,identity))
            write();reader=OriginalGguf(manifest,identity)
            p=Path(files[0]['path']);p.write_bytes(p.read_bytes()+b'x');reject('source_change_before_row_read',lambda:reader.rows('synthetic.0',[0]))
    return {'synthetic_reader_controls':rejected,'synthetic_reader_positive':True,'actual_model_reads':False,'role_validation_separately_tested':True}


if __name__=='__main__':print(json.dumps(run(),sort_keys=True))
