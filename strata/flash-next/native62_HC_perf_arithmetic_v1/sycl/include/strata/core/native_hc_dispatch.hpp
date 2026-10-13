// Shared exact-source dispatch: ordinary decode, prompt chunks and verifier rows.
#pragma once
#include <cstdlib>
#include <cstring>
#include "strata/core/weights.hpp"
#include "strata/core/native_hc_policy.hpp"
#include "strata/kernels/hc_native_composition.hpp"
namespace strata::core {
inline bool native_hc_bind(const WeightRef* wn,const WeightRef* wd,const WeightRef* wu,const WeightRef* wi,
                           strata::kernels::HcNativeArgs& a,std::string& err) {
    if (!native_hc_policy(err)) return false;
    if (!wn || !wd || !wu || wn->hc_source_type!=0 || wd->hc_source_type!=8 || wu->hc_source_type!=8 ||
        wn->hc_source_ne0!=10240 || wn->hc_source_ne1!=1 || wn->hc_source_rank!=1 ||
        wd->hc_source_ne0!=10240 || wd->hc_source_ne1!=320 || wd->hc_source_rank!=2 ||
        wu->hc_source_ne0!=320 || wu->hc_source_ne1!=10240 || wu->hc_source_rank!=2 ||
        (wi && (wi->hc_source_type!=0 || wi->hc_source_ne0!=10240 || wi->hc_source_ne1!=4 || wi->hc_source_rank!=2))) {
        err="native HC: missing or incompatible exact-source view"; return false;
    }
    a.w={};
    a.w.norm=wn->hc_source_f32; a.w.norm_floats=wn->hc_source_bytes/4;
    a.w.down=wd->hc_source_q8; a.w.down_bytes=wd->hc_source_bytes;
    a.w.up=wu->hc_source_q8; a.w.up_bytes=wu->hc_source_bytes;
    if (wi) { a.w.inject=wi->hc_source_f32; a.w.inject_floats=wi->hc_source_bytes/4; }
    return true;
}
inline bool native_hc_submit(const strata::kernels::HcNativeArgs& a,void* stream,std::string& err) {
#ifdef STRATA_SYCL_Q8_HC_BUILT
    if (!strata::kernels::hc_native_read_f32(a,stream)) { err="native HC: composed read rejected"; return false; }
    return true;
#else
    err="native HC: kernel build disabled"; return false;
#endif
}
inline bool native_hc_write_rows(float* R,const float* bo,const float* injection,int rows,void* stream,std::string& err) {
#ifdef STRATA_SYCL_Q8_HC_BUILT
    // Model owners establish contiguous buffers. This helper chunks only known
    // contiguous token-major arrays, never pointer arrays or inferred strides.
    if (rows<1 || !R || !bo || !injection) { err="native HC: empty or null residual write"; return false; }
    for (int t=0;t<rows;t+=8) {
        const int n=rows-t<8?rows-t:8;
        if (!strata::kernels::hc_native_write_f32(R+size_t(t)*10240,size_t(n)*10240,
            bo+size_t(t)*2560,size_t(n)*2560,injection+size_t(t)*4,size_t(n)*4,n,stream)) {
            err="native HC: residual write rejected"; return false;
        }
    }
    return true;
#else
    err="native HC: kernel build disabled"; return false;
#endif
}
// Default OFF. Only prefill callers consult this selection; decode is unchanged.
inline int native_hc_prefill_coalesce() {
    static const int value=[] {const char* v=std::getenv("STRATA_HC_PREFILL_COALESCE");
        if(!v || std::strcmp(v,"0")==0)return 0;
        if(std::strcmp(v,"1")==0)return 1;
        return -1;}();
    return value;
}
inline bool native_hc_prefill_submit(const strata::kernels::HcNativeArgs& a,void* stream,std::string& err) {
#ifdef STRATA_SYCL_Q8_HC_BUILT
    if(!strata::kernels::hc_native_prefill_read_f32(a,stream)){err="native HC: prefill composed read rejected";return false;}
    return true;
#else
    err="native HC: kernel build disabled";return false;
#endif
}
inline bool native_hc_prefill_write_rows(float* R,const float* bo,const float* injection,int rows,void* stream,std::string& err) {
    const int selected=native_hc_prefill_coalesce();
    if(selected<0){err="native HC: invalid prefill coalescing flag";return false;}
    if(!selected)return native_hc_write_rows(R,bo,injection,rows,stream,err);
#ifdef STRATA_SYCL_Q8_HC_BUILT
    if(rows<1 || !R || !bo || !injection){err="native HC: empty or null prefill write";return false;}
    for(int t=0;t<rows;t+=512){const int n=rows-t<512?rows-t:512;
        if(!strata::kernels::hc_native_prefill_write_f32(R+size_t(t)*10240,size_t(n)*10240,
            bo+size_t(t)*2560,size_t(n)*2560,injection+size_t(t)*4,size_t(n)*4,n,stream)){
            err="native HC: prefill residual write rejected";return false;}}
    return true;
#else
    err="native HC: kernel build disabled";return false;
#endif
}

}
