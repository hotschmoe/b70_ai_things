// Synthetic current-SYCL component oracle. No model payload or captured activation inputs.
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/kernels/verify_kernels.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
constexpr int S=128, HK=16, HV=48, C=10240, V=S*HV, STEPS=3;
constexpr size_t ST=size_t(S)*HV*S, CV=size_t(C)*3;
constexpr float EPS=1e-6f;
struct Buffers {
    sycl::queue& q; std::vector<void*> owned;
    explicit Buffers(sycl::queue& queue):q(queue) {}
    template<class T> T* make(size_t n) {
        auto p=sycl::malloc_device<T>(n,q); if(!p) throw std::runtime_error("USM allocation failed");
        owned.push_back(p); return p;
    }
    ~Buffers() { q.wait_and_throw(); for(auto p:owned) sycl::free(p,q); }
};
static float synthetic(size_t i,int salt,float scale) {
    return float(int((i*37+size_t(salt)*101)%257)-128)*scale;
}
static bool equal(const std::vector<float>& a,const std::vector<float>& b) {
    return a.size()==b.size() && std::memcmp(a.data(),b.data(),a.size()*4)==0;
}
static void save(const fs::path& dir,const std::string& name,const std::vector<float>& x) {
    const auto path=dir/(name+".f32"); if(fs::exists(path)) throw std::runtime_error("Refusing raw artifact overwrite");
    std::ofstream f(path,std::ios::binary);f.write(reinterpret_cast<const char*>(x.data()),x.size()*4);
    if(!f) throw std::runtime_error("Raw artifact write failed");
}
static std::vector<float> readback(sycl::queue& q,const float* p,size_t n) {
    std::vector<float> x(n);q.memcpy(x.data(),p,n*4).wait_and_throw();return x;
}
static bool comparison(const char* case_name,const char* component,const std::vector<float>& a,const std::vector<float>& b) {
    size_t bad=0,first=a.size();double maxerr=0;
    if(a.size()!=b.size()) throw std::runtime_error("Comparison shape mismatch");
    for(size_t i=0;i<a.size();++i) {
        if(!std::isfinite(a[i])||!std::isfinite(b[i])) throw std::runtime_error("Nonfinite component");
        if(std::memcmp(&a[i],&b[i],4)!=0) {++bad;if(first==a.size())first=i;}
        maxerr=std::max(maxerr,std::abs(double(a[i])-b[i]));
    }
    std::printf("GDN35_COMPARE case=%s component=%s words=%zu differing=%zu first=%zu max_abs=%.17g bitwise=%d\n",case_name,component,a.size(),bad,first,maxerr,bad==0);
    return bad==0;
}
// Independent scalar FP64 recurrence diagnostic, native state layout [row][head][column].
// No captured native state is supplied. This does not claim native intrinsic/reduction parity.
static std::vector<double> scalar_conv(const std::vector<float>& x,const std::vector<float>& weights,int n) {
    std::vector<double> h(size_t(n)*C,0.0);
    for(int t=0;t<n;++t) {
        for(int c=0;c<C;++c) {
            double sum=0;
            for(int tap=0;tap<4;++tap) {const int src=t+tap-3;if(src>=0)sum+=double(x[size_t(src)*C+c])*weights[c*4+tap];}
            h[size_t(t)*C+c]=sum/(1.0+std::exp(-sum));
        }
        for(int head=0;head<2*HK;++head) {
            double sq=0;for(int row=0;row<S;++row){double v=h[size_t(t)*C+head*S+row];sq+=v*v;}
            const double inv=1.0/std::sqrt(sq+double(EPS));
            for(int row=0;row<S;++row)h[size_t(t)*C+head*S+row]*=inv;
        }
    }
    return h;
}
static std::vector<double> scalar_state(const std::vector<double>& h,const std::vector<float>& gate,
                                        const std::vector<float>& beta,int n) {
    std::vector<double> st(ST,0.0);
    for(int t=0;t<n;++t) for(int head=0;head<HV;++head) for(int col=0;col<S;++col) {
        const int kh=head%HK;const double decay=std::exp(double(gate[t*HV+head]));double sk=0;
        for(int row=0;row<S;++row) {const size_t i=(size_t(row)*HV+head)*S+col;st[i]*=decay;sk+=st[i]*double(h[size_t(t)*C+S*HK+kh*S+row]);}
        const double delta=(double(h[size_t(t)*C+2*S*HK+head*S+col])-sk)*double(beta[t*HV+head]);
        for(int row=0;row<S;++row) st[(size_t(row)*HV+head)*S+col]+=double(h[size_t(t)*C+S*HK+kh*S+row])*delta;
    }
    return st;
}
int main(int argc,char** argv) try {
    if(argc!=3||std::string(argv[1])!="--output") throw std::runtime_error("usage: gdn_state_transition35_v1 --output NEW_DIRECTORY");
    if(std::getenv("STRATA_VERIFY_EAGER")) throw std::runtime_error("Eager must be absent even0");
    if(const char* p=std::getenv("STRATA_GDN_SPLIT");p&&std::string(p)!="0") throw std::runtime_error("This initial normal-path fixture requires GDN_SPLIT absent or0");
    fs::path out(argv[2]);if(fs::exists(out))throw std::runtime_error("Output directory must be new");fs::create_directories(out);
    auto& q=dpct::get_in_order_queue();
    std::printf("GDN35_CONFIG synthetic=1 model_math_qualified=0 state_f32=1 S=%d h_k=%d h_v=%d channels=%d steps=%d device=%s\n",S,HK,HV,C,STEPS,q.get_device().get_info<sycl::info::device::name>().c_str());
    int failures=0,comparisons=0;
    {
    Buffers mem(q);
    std::vector<float> qkv(STEPS*C),cw(C*4),gate(STEPS*HV),beta(STEPS*HV),z(STEPS*V),gamma(S),zero_state(ST,0),zero_conv(CV,0);
    for(size_t i=0;i<qkv.size();++i)qkv[i]=synthetic(i,3,0.001f);
    for(size_t i=0;i<cw.size();++i)cw[i]=synthetic(i,7,0.002f);
    for(size_t i=0;i<gate.size();++i) {gate[i]=-0.5f-float(i%17)*0.1f;beta[i]=0.1f+float(i%11)*0.05f;}
    for(size_t i=0;i<z.size();++i)z[i]=synthetic(i,13,0.005f);
    for(size_t i=0;i<gamma.size();++i)gamma[i]=1.0f+float(i%7)*0.01f;
    save(out,"input-qkv",qkv);save(out,"input-conv-weights",cw);save(out,"input-gate",gate);save(out,"input-beta",beta);save(out,"input-z",z);save(out,"input-gamma",gamma);
    auto upload=[&](const std::vector<float>& x) {auto p=mem.make<float>(x.size());q.memcpy(p,x.data(),x.size()*4).wait_and_throw();return p;};
    auto dq=upload(qkv),dw=upload(cw),dg=upload(gate),db=upload(beta),dz=upload(z),dn=upload(gamma);
    auto sr=mem.make<float>(ST),cr=mem.make<float>(CV),hr=mem.make<float>(STEPS*C),yr=mem.make<float>(STEPS*V);
    auto sw=mem.make<float>(ST),cwst=mem.make<float>(CV),hw=mem.make<float>(STEPS*C),yw=mem.make<float>(STEPS*V),scratch=mem.make<float>(STEPS*V);
    auto nk=mem.make<int32_t>(1);auto one=mem.make<int32_t>(1);int32_t one_host=1;q.memcpy(one,&one_host,4).wait_and_throw();
    auto reset=[&](float* st,float* cv) {q.memset(st,0,ST*4);q.memset(cv,0,CV*4);q.wait_and_throw();};
    auto t1=[&](float* st,float* cv,float* h,float* y,int t) {
        strata::kernels::gdn_conv_l2_multi(cv,dq+size_t(t)*C,dw,h+size_t(t)*C,C,2*HK,EPS,1,&q,0,true);
        strata::kernels::gdn_step_norm_multi(st,h+size_t(t)*C,C,dg+t*HV,db+t*HV,dz+size_t(t)*V,dn,EPS,y+size_t(t)*V,HK,HV,1,one,&q,0,nullptr);
        q.wait_and_throw();
    };
    reset(sr,cr);
    std::vector<std::vector<float>> states={zero_state},histories={zero_conv};
    for(int t=0;t<STEPS;++t) {t1(sr,cr,hr,yr,t);states.push_back(readback(q,sr,ST));histories.push_back(readback(q,cr,CV));save(out,"t1-state-after-"+std::to_string(t+1),states.back());save(out,"t1-conv-after-"+std::to_string(t+1),histories.back());}
    auto h_ref=readback(q,hr,STEPS*C),y_ref=readback(q,yr,STEPS*V);save(out,"t1-normalized-qkv",h_ref);save(out,"t1-output",y_ref);
    // T2 forward must leave full F32 persistent state/history untouched. Commit accepts0/1/2 rows.
    for(int keep=0;keep<=2;++keep) {
        std::string label="t2-keep"+std::to_string(keep);reset(sw,cwst);q.memset(yw,0,STEPS*V*4).wait_and_throw();
        strata::kernels::gdn_conv_l2_multi(cwst,dq,dw,hw,C,2*HK,EPS,2,&q,0,false);
        strata::kernels::gdn_step_norm_multi(sw,hw,C,dg,db,dz,dn,EPS,yw,HK,HV,2,nullptr,&q,0,nullptr);q.wait_and_throw();
        auto check=[&](const char* component,const std::vector<float>& a,const std::vector<float>& b) {++comparisons;if(!comparison(label.c_str(),component,a,b))++failures;};
        auto forward_state=readback(q,sw,ST),forward_conv=readback(q,cwst,CV),forward_h=readback(q,hw,2*C),forward_y=readback(q,yw,2*V);
        save(out,label+"-forward-state",forward_state);save(out,label+"-forward-conv",forward_conv);save(out,label+"-forward-normalized-qkv",forward_h);save(out,label+"-forward-output",forward_y);
        check("forward-state-unmodified",forward_state,zero_state);check("forward-conv-unmodified",forward_conv,zero_conv);
        check("forward-normalized-qkv",forward_h,std::vector<float>(h_ref.begin(),h_ref.begin()+2*C));
        check("forward-output",forward_y,std::vector<float>(y_ref.begin(),y_ref.begin()+2*V));
        int32_t keep_host=keep;q.memcpy(nk,&keep_host,4).wait_and_throw();
        strata::kernels::gdn_conv_commit(cwst,dq,C,nk,&q);
        strata::kernels::gdn_step_norm_multi(sw,hw,C,dg,db,dz,dn,EPS,scratch,HK,HV,2,nk,&q,2,nullptr);q.wait_and_throw();
        auto committed=readback(q,sw,ST),history=readback(q,cwst,CV);
        save(out,label+"-committed-state",committed);save(out,label+"-committed-conv",history);
        check("accepted-state",committed,states[keep]);check("accepted-conv",history,histories[keep]);
        t1(sw,cwst,hw,yw,keep);
        check("carry-state",readback(q,sw,ST),states[keep+1]);check("carry-conv",readback(q,cwst,CV),histories[keep+1]);
        check("carry-output",readback(q,yw+size_t(keep)*V,V),std::vector<float>(y_ref.begin()+size_t(keep)*V,y_ref.begin()+size_t(keep+1)*V));
        save(out,label+"-carry-state",readback(q,sw,ST));save(out,label+"-carry-conv",readback(q,cwst,CV));save(out,label+"-carry-output",readback(q,yw+size_t(keep)*V,V));
    }
    // Deliberately wrong persistent state must be observable in the next carried output.
    q.memcpy(sw,states[2].data(),ST*4);q.memcpy(cwst,histories[2].data(),CV*4);float modified=states[2][0]+1.0f;q.memcpy(sw,&modified,4).wait_and_throw();
    t1(sw,cwst,hw,yw,2);auto negative=readback(q,yw+2*V,V);save(out,"negative-modified-state-output",negative);
    bool rejected=!equal(negative,std::vector<float>(y_ref.begin()+2*V,y_ref.end()));if(!rejected)++failures;
    std::printf("GDN35_NEGATIVE modified_state_output_detected=%d\n",rejected);
    auto own_h=scalar_conv(qkv,cw,STEPS);auto scalar=scalar_state(own_h,gate,beta,STEPS);double err=0,den=0,maxabs=0;
    for(size_t i=0;i<ST;++i){double d=states[STEPS][i]-scalar[i];err+=d*d;den+=scalar[i]*scalar[i];maxabs=std::max(maxabs,std::abs(d));}
    std::printf("GDN35_INDEPENDENT_DIAGNOSTIC fp64_state_nmse=%.17g max_abs=%.17g numerical_tolerance_qualified=0\n",err/(den+1e-300),maxabs);
    }
    q.wait_and_throw();
    std::printf("GDN35_RESULT comparisons=%d failures=%d synthetic_component_passed=%d model_math_qualified=0 all_owned_allocations_freed=1\n",comparisons,failures,failures==0);
    return failures?1:0;
} catch(const std::exception& e) {std::fprintf(stderr,"GDN35_ERROR %s\n",e.what());return 2;}
