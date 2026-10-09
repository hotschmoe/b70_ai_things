// Native standalone/pending residual write and actual shared-wrapper qualification.
// GPU execution requires bin/gpu-run and STRATA_SYCL_Q8_HC_BUILT=1 at compile.
#define main hc_projection_fixture_unused_main
#include "native_q8_hc_gpu_fixture.cpp"
#undef main
#include "strata/kernels/hc_native_composition.hpp"
#ifndef HC_FIXTURE_CPU_ONLY
#include "strata/core/native_hc_dispatch.hpp"
#endif
namespace {
constexpr size_t write_D=10240,write_N=2560,write_H=4,write_L=320;
const char* write_profiles[]={"tiny","mixed","saturated"};
struct WriteInput {
    int rows,profile; std::vector<float> residual,bo,inject;std::vector<double> reference;
    WriteInput(int t,int p):rows(t),profile(p),residual(size_t(t)*write_D),bo(size_t(t)*write_N),inject(size_t(t)*write_H),reference(size_t(t)*write_D) {
        for(size_t i=0;i<residual.size();++i) residual[i]=float(int((i*31+i/write_D*17)%251)-125)*0.031251f;
        for(size_t i=0;i<bo.size();++i) bo[i]=float(int((i*19+i/write_N*29)%127)-63)*0.007813f;
        for(size_t i=0;i<inject.size();++i) inject[i]=p==0 ? float(int(i%7)-3)*0.000001f : p==1 ? float(int((i*17)%31)-15)*0.250123f : (i%2 ? -96.001f:96.001f);
        for(int t0=0;t0<t;++t0) for(size_t c=0;c<write_H;++c) for(size_t d=0;d<write_N;++d) {
            size_t i=size_t(t0)*write_D+c*write_N+d;
            double weight=2.0/(1.0+std::exp(-double(inject[size_t(t0)*write_H+c])/4.0));
            reference[i]=double(residual[i])+double(bo[size_t(t0)*write_N+d])*weight;
        }
    }
};
Numeric write_metric(const WriteInput& in,const std::vector<float>& out) {
    Numeric n;double e2=0,r2=0,me=0,mr=0;
    for(size_t i=0;i<out.size();++i) {n.finite &= std::isfinite(out[i]);double e=double(out[i])-in.reference[i];e2+=e*e;r2+=in.reference[i]*in.reference[i];me=std::max(me,std::abs(e));mr=std::max(mr,std::abs(in.reference[i]));}
    n.nmse=e2/std::max(1e-30,r2);n.linf=me/std::max(1e-6,mr);return n;
}
#ifndef HC_FIXTURE_CPU_ONLY
float* fptr(DeviceBuffer& b) {return reinterpret_cast<float*>(b.data());}
struct WriteBuffers {
    sycl::queue& q;const WriteInput& input;DeviceBuffer R,bo,inject;
    WriteBuffers(sycl::queue& Q,const WriteInput& in):q(Q),input(in),R(q,in.residual.data(),in.residual.size()*4),bo(q,in.bo.data(),in.bo.size()*4),inject(q,in.inject.data(),in.inject.size()*4) {}
    void run(int chunk,bool wrapper=false) {
        q.memcpy(R.data(),input.residual.data(),input.residual.size()*4).wait_and_throw();
        if(wrapper) {std::string err;if(!strata::core::native_hc_write_rows(fptr(R),fptr(bo),fptr(inject),input.rows,&q,err)) throw std::runtime_error(err);}
        else for(int t=0;t<input.rows;t+=chunk) {int n=std::min(chunk,input.rows-t);
            if(!strata::kernels::hc_native_write_f32(fptr(R)+size_t(t)*write_D,size_t(n)*write_D,fptr(bo)+size_t(t)*write_N,size_t(n)*write_N,fptr(inject)+size_t(t)*write_H,size_t(n)*write_H,n,&q)) throw std::runtime_error("valid standalone write rejected");
        }
        q.wait_and_throw();
    }
    bool immutable_inputs() {return bo.valid(bo.read(),input.bo.data()) && inject.valid(inject.read(),input.inject.data());}
    bool guarded_immutable() {return R.valid(R.read()) && immutable_inputs();}
};
std::vector<uint8_t> pending_result(sycl::queue& q,const WriteInput& input,const Fixture& dw,const Fixture& uw,bool& guards,bool& immutable) {
    std::vector<float> norm(write_D,1.000123f);
    DeviceBuffer R(q,input.residual.data(),input.residual.size()*4),bo(q,input.bo.data(),input.bo.size()*4),inject(q,input.inject.data(),input.inject.size()*4),
        wn(q,norm.data(),norm.size()*4),wd(q,dw.weights.data(),dw.weights.size()),wu(q,uw.weights.data(),uw.weights.size()),
        out(q,nullptr,input.rows*write_D*4),xn(q,nullptr,input.rows*write_D*4),lo(q,nullptr,input.rows*write_L*4),gate(q,nullptr,input.rows*write_D*4),
        rs(q,nullptr,input.rows*write_H*4),mixed(q,nullptr,input.rows*write_N*4);
    for(int t=0;t<input.rows;t+=8) {
        size_t n=size_t(std::min(8,input.rows-t)),tb=size_t(t);strata::kernels::HcNativeArgs a;a.tokens=int(n);a.apply=true;
        a.w.norm=fptr(wn);a.w.norm_floats=write_D;a.w.down=wd.data();a.w.down_bytes=dw.weights.size();a.w.up=wu.data();a.w.up_bytes=uw.weights.size();
        a.R=fptr(R)+tb*write_D;a.R_floats=n*write_D;a.bo_prev=fptr(bo)+tb*write_N;a.bo_floats=n*write_N;a.inj_prev=fptr(inject)+tb*write_H;a.prev_floats=n*write_H;
        a.R_out=fptr(out)+tb*write_D;a.R_out_floats=n*write_D;a.xn=fptr(xn)+tb*write_D;a.xn_floats=n*write_D;a.lo=fptr(lo)+tb*write_L;a.lo_floats=n*write_L;
        a.gate=fptr(gate)+tb*write_D;a.gate_floats=n*write_D;a.rs=fptr(rs)+tb*write_H;a.rs_floats=n*write_H;a.mixed=fptr(mixed)+tb*write_N;a.mixed_floats=n*write_N;
        if(!strata::kernels::hc_native_read_f32(a,&q)) throw std::runtime_error("pending composition rejected");
    }
    q.wait_and_throw();for(auto* b:{&R,&bo,&inject,&wn,&wd,&wu,&out,&xn,&lo,&gate,&rs,&mixed}) guards &= b->valid(b->read());
    immutable &= R.valid(R.read(),input.residual.data()) && bo.valid(bo.read(),input.bo.data()) && inject.valid(inject.read(),input.inject.data()) && wn.valid(wn.read(),norm.data()) && wd.valid(wd.read(),dw.weights.data()) && wu.valid(wu.read(),uw.weights.data());
    return out.read();
}
int write_rejections(sycl::queue& q) {
    WriteInput in(1,1);WriteBuffers b(q,in);auto initial=b.R.read();int count=0;
    auto reject=[&](float* r,size_t nr,const float* bo,size_t nb,const float* injection,size_t ni,int rows,void* queue) {
        if(strata::kernels::hc_native_write_f32(r,nr,bo,nb,injection,ni,rows,queue)) throw std::runtime_error("invalid write accepted");++count;
    };
    reject(nullptr,write_D,fptr(b.bo),write_N,fptr(b.inject),write_H,1,&q);
    reject(fptr(b.R),write_D,nullptr,write_N,fptr(b.inject),write_H,1,&q);
    reject(fptr(b.R),write_D,fptr(b.bo),write_N,nullptr,write_H,1,&q);
    reject(fptr(b.R),write_D-1,fptr(b.bo),write_N,fptr(b.inject),write_H,1,&q);
    reject(fptr(b.R),write_D,fptr(b.bo),write_N-1,fptr(b.inject),write_H,1,&q);
    reject(fptr(b.R),write_D,fptr(b.bo),write_N,fptr(b.inject),write_H-1,1,&q);
    reject(fptr(b.R),write_D,fptr(b.bo),write_N,fptr(b.inject),write_H,0,&q);
    reject(fptr(b.R),write_D,fptr(b.bo),write_N,fptr(b.inject),write_H,-1,&q);
    reject(fptr(b.R),9*write_D,fptr(b.bo),9*write_N,fptr(b.inject),9*write_H,9,&q);
    sycl::queue unordered(q.get_context(),q.get_device());reject(fptr(b.R),write_D,fptr(b.bo),write_N,fptr(b.inject),write_H,1,&unordered);
    auto wrapper_reject=[&](float* r,const float* bo,const float* injection,int rows) {std::string err;if(strata::core::native_hc_write_rows(r,bo,injection,rows,&q,err)) throw std::runtime_error("invalid wrapper accepted");++count;};
    wrapper_reject(nullptr,fptr(b.bo),fptr(b.inject),1);wrapper_reject(fptr(b.R),nullptr,fptr(b.inject),1);wrapper_reject(fptr(b.R),fptr(b.bo),nullptr,1);
    wrapper_reject(fptr(b.R),fptr(b.bo),fptr(b.inject),0);wrapper_reject(fptr(b.R),fptr(b.bo),fptr(b.inject),-1);
    q.wait_and_throw();unordered.wait_and_throw();if(initial!=b.R.read() || !b.guarded_immutable()) throw std::runtime_error("rejected write changed buffers");
    return count;
}
#endif
}
int main(int argc,char** argv) {
    try {
        std::ofstream receipt;std::ostream* log=&std::cout;if(argc>2) throw std::runtime_error("usage: write_fixture [receipt.jsonl]");
        if(argc==2) {receipt.open(argv[1]);if(!receipt) throw std::runtime_error("cannot open receipt");log=&receipt;}*log<<std::setprecision(17)<<std::boolalpha;
        int cases=0,failed=0,rejected=0,negatives=0;bool guard_negative=false;
#ifndef HC_FIXTURE_CPU_ONLY
        sycl::queue q(sycl::gpu_selector_v,sycl::property::queue::in_order{});std::cerr<<"Selected GPU: "<<q.get_device().get_info<sycl::info::device::name>()<<'\n';
        rejected=write_rejections(q);Fixture dw(write_D,write_L,1,0,true),uw(write_L,write_D,1,0,true);
        DeviceBuffer guard_probe(q,nullptr,4);auto corrupt_guard=guard_probe.read();corrupt_guard.back()=0;guard_negative=!guard_probe.valid(corrupt_guard);if(!guard_negative) throw std::runtime_error("write guard negative control accepted");
#endif
        for(int rows:{1,2,4,8,9,16,17}) for(int profile:{0,1,2}) {
            WriteInput input(rows,profile);std::vector<float> values(input.reference.size());bool repeat=true,single=true,wrapper=true,pending=true,guards=true,immutable=true;
#ifndef HC_FIXTURE_CPU_ONLY
            WriteBuffers a(q,input);a.run(8);auto first=a.R.read();std::memcpy(values.data(),first.data()+guard,values.size()*4);guards &= a.guarded_immutable();
            WriteInput alternate=input;for(float& v:alternate.bo) v=v*0.5f+0.000123f;WriteBuffers b(q,alternate);b.run(8);
            a.run(8);repeat=first==a.R.read();guards &= a.guarded_immutable() && b.guarded_immutable();
            WriteBuffers singles(q,input);singles.run(1);single=first==singles.R.read();guards &= singles.guarded_immutable();
            WriteBuffers shared(q,input);shared.run(8,true);wrapper=first==shared.R.read();guards &= shared.guarded_immutable();
            immutable &= a.immutable_inputs() && b.immutable_inputs() && singles.immutable_inputs() && shared.immutable_inputs();
            pending=first==pending_result(q,input,dw,uw,guards,immutable);
#else
            for(size_t i=0;i<values.size();++i) values[i]=float(input.reference[i]);
#endif
            Numeric n=write_metric(input,values);double mr=0;for(double v:input.reference) mr=std::max(mr,std::abs(v));auto corrupt=values;corrupt[0]=float(input.reference[0]+0.01*std::max(1e-6,mr));Numeric bad=write_metric(input,corrupt);
            if(bad.finite && bad.nmse<=1e-6 && bad.linf<=1e-4) throw std::runtime_error("write numeric negative control accepted");
            ++negatives;
            bool pass=n.finite && n.nmse<=1e-6 && n.linf<=1e-4 && repeat && single && wrapper && pending && guards && immutable;if(!pass) ++failed;
            *log<<"{\"kind\":\"case\",\"rows\":"<<rows<<",\"profile\":\""<<write_profiles[profile]<<"\",\"finite\":"<<n.finite<<",\"nmse\":";if(std::isfinite(n.nmse)) *log<<n.nmse;else *log<<"null";
            *log<<",\"normalized_linf\":";if(std::isfinite(n.linf)) *log<<n.linf;else *log<<"null";
            *log<<",\"exact_repeat\":"<<repeat<<",\"exact_batch_vs_single\":"<<single<<",\"exact_shared_wrapper\":"<<wrapper<<",\"exact_pending_write\":"<<pending<<",\"guards\":"<<guards<<",\"unchanged_inputs\":"<<immutable<<",\"numeric_negative_control\":true,\"pass\":"<<pass<<"}\n";log->flush();++cases;
        }
#ifdef HC_FIXTURE_CPU_ONLY
        const char* mode="CPU_REFERENCE_ONLY";bool pass=cases==21 && failed==0 && negatives==21;
#else
        const char* mode="GPU_WRITE_NUMERICAL";bool pass=cases==21 && failed==0 && negatives==21 && rejected==15 && guard_negative;
#endif
        *log<<"{\"kind\":\"summary\",\"mode\":\""<<mode<<"\",\"cases\":"<<cases<<",\"failed\":"<<failed<<",\"rejection_cases\":"<<rejected<<",\"numeric_negative_controls\":"<<negatives<<",\"guard_negative_control\":"<<guard_negative<<",\"nmse_gate\":1e-6,\"normalized_linf_gate\":1e-4,\"pass\":"<<pass<<"}\n";
        return pass ? 0:1;
    }catch(const std::exception& e){std::cerr<<"Write fixture failure: "<<e.what()<<'\n';return 2;}
}
