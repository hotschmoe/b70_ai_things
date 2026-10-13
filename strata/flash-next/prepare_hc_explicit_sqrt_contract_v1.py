"""Source-only default-OFF explicit sqrt/divide on retained HC56 composition.
Perf merge remains a separate prerequisite; no frozen SDK/engine is modified.
"""
import argparse,difflib,hashlib,json
from pathlib import Path
BASE=Path('/mnt/vm_8tb/b70/build/strata-native56-incremental55-v3-20261012T055750Z-ophmtttx/source/sycl/src/kernels/hc_native_composition.cpp')
BASE_SHA='142ffa2f2cad09e16efae95b85c3d8d783d2fbc6897c99f4715335e74e5b194b'
FLAG='STRATA_HC_EXPLICIT_SQRT_DIVIDE'
def require(ok,msg):
 if not ok:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def transform(text):
 require(hashlib.sha256(text.encode('ascii')).hexdigest()==BASE_SHA,'Exact optimized56 HC composition required')
 anchor='float sigmoid(float x) { return 1.0f/(1.0f+sycl::exp(-x)); }\n'
 helper='''// Independent explicit two-operation RMS contract, default OFF.
int hc_explicit_sqrt_divide() {
    static const int mode=[] {const char* v=std::getenv("STRATA_HC_EXPLICIT_SQRT_DIVIDE");
        if(!v || std::strcmp(v,"0")==0)return 0;
        if(std::strcmp(v,"1")==0)return 1;
        return -1;}();
    return mode;
}
'''
 entry='static bool hc_read_impl(const HcNativeArgs& a, void *stream,bool prefill) {\n';guard='    const int explicit_sqrt_divide=hc_explicit_sqrt_divide();\n    if(explicit_sqrt_divide<0)return false; // Refuse before any HC queue work.\n'
 old='        const float rs=sycl::rsqrt(reduce32(item.get_sub_group(),sum)/float(N)+a.eps);'
 new='''        const float rs=explicit_sqrt_divide
            ? 1.0f/sycl::sqrt(reduce32(item.get_sub_group(),sum)/float(N)+a.eps)
            : sycl::rsqrt(reduce32(item.get_sub_group(),sum)/float(N)+a.eps);'''
 write='static bool hc_write_impl(float* R,size_t R_floats,const float* bo,size_t bo_floats,\n                        const float* inject,size_t inject_floats,int tokens,void* stream,bool prefill) {\n'
 writeguard='    if(hc_explicit_sqrt_divide()<0)return false; // Same invalid-option refusal.\n'
 for a in(anchor,entry,old,write):require(text.count(a)==1,'Unique source contract anchor required')
 result=text.replace(anchor,anchor+helper).replace(entry,entry+guard).replace(old,new).replace(write,write+writeguard)
 undo=result.replace(anchor+helper,anchor).replace(entry+guard,entry).replace(new,old).replace(write+writeguard,write)
 require(undo==text,'Exact arithmetic-only source undo failed');return result

def prepare(out):
 require(not out.exists(),'New arithmetic source proposal output required');original=BASE.read_text(encoding='ascii');result=transform(original);out.mkdir();(out/'hc_native_composition.cpp').write_text(result,encoding='ascii');patch=''.join(difflib.unified_diff(original.splitlines(True),result.splitlines(True),fromfile='a/sycl/src/kernels/hc_native_composition.cpp',tofile='b/sycl/src/kernels/hc_native_composition.cpp'));(out/'hc-explicit-sqrt-divide.patch').write_text(patch,encoding='ascii');receipt={'schema':1,'source_only':True,'base_path':str(BASE),'base_sha256':BASE_SHA,'result_sha256':sha(out/'hc_native_composition.cpp'),'patch_sha256':sha(out/'hc-explicit-sqrt-divide.patch'),'flag':FLAG,'default':0,'values':['0','1'],'invalid_refuses_before_queue_work':True,'host_parser_cached_per_process':True,'public_layout_changed':False,'projection_tile_and_prefill_coalesce_retained':True,'additional_kernels_queries_waits_copies':False,'compiled':False,'GPU_executed':False};(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='ascii');return receipt
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);prepare(p.parse_args().output)
