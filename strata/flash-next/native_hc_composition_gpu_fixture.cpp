// Composed HC numerical qualification. GPU execution requires bin/gpu-run.
// Reuse the independently tested exact-Q8 fixture decoder and guarded buffers.
#define main hc_projection_fixture_unused_main
#include "native_q8_hc_gpu_fixture.cpp"
#undef main
#include "strata/kernels/hc_native_composition.hpp"
#include <memory>
namespace {
constexpr size_t D=10240,N=2560,H=4,L=320;
struct CompositionInput {
    int tokens; bool apply, injection, inplace; int profile;
    std::vector<float> residual, norm, bo, prior;
    CompositionInput(int t,bool a,bool j,bool ip,int p) : tokens(t),apply(a),injection(j),inplace(ip),profile(p),
        residual(t*D),norm(D),bo(a ? t*N:0),prior(a ? t*H:0) {
        for (size_t i=0;i<norm.size();++i) norm[i]=0.800123f+float(i%17)*0.019531f;
        float amplitude=p ? 0.00001f:0.03f;
        for (size_t i=0;i<residual.size();++i) residual[i]=float(int((i*31+i/D*37)%251)-125)*amplitude;
        for (size_t i=0;i<bo.size();++i) bo[i]=float(int((i*17)%127)-63)*amplitude*0.2f;
        for (size_t i=0;i<prior.size();++i) prior[i]=float(int((i*13)%23)-11)*0.5f+0.001f;
    }
};
double logistic(double x) { return 1.0/(1.0+std::exp(-x)); }
std::vector<double> project_reference(const Fixture& weight,const std::vector<double>& x,int tokens) {
    std::vector<double> y(size_t(tokens)*weight.m);
    for (int t=0;t<tokens;++t) for(int r=0;r<weight.m;++r) {
        double sum=0;
        for(int c=0;c<weight.k;++c) {
            double w;
            if(weight.q8) {
                size_t off=(size_t(r)*(weight.k/32)+c/32)*34;
                uint16_t bits=uint16_t(weight.weights[off])|(uint16_t(weight.weights[off+1])<<8);
                int code=weight.weights[off+2+c%32];
                w=scale_decode(bits)*(code<128 ? code:code-256);
            } else { float f; std::memcpy(&f,weight.weights.data()+(size_t(r)*weight.k+c)*4,4); w=f; }
            sum+=w*x[size_t(t)*weight.k+c];
        }
        y[size_t(t)*weight.m+r]=sum;
    }
    return y;
}
struct Oracle {
    std::vector<double> residual,rs,xn,down,lo,gate,inject,mixed;
    Oracle(const CompositionInput& in,const Fixture& dw,const Fixture& uw,const Fixture& iw) :
        residual(in.residual.begin(),in.residual.end()),rs(in.tokens*H),xn(in.tokens*D),mixed(in.tokens*N) {
        if(in.apply) for(int t=0;t<in.tokens;++t) for(size_t c=0;c<H;++c) for(size_t d=0;d<N;++d) {
            size_t i=size_t(t)*D+c*N+d;
            residual[i]+=double(in.bo[size_t(t)*N+d])*2*logistic(double(in.prior[size_t(t)*H+c])/4);
        }
        for(int t=0;t<in.tokens;++t) for(size_t c=0;c<H;++c) {
            double sum=0; for(size_t d=0;d<N;++d) { double r=residual[size_t(t)*D+c*N+d]; sum+=r*r; }
            double inv=1/std::sqrt(sum/N+double(1e-6f)); rs[size_t(t)*H+c]=inv;
            for(size_t d=0;d<N;++d) { size_t i=c*N+d; xn[size_t(t)*D+i]=residual[size_t(t)*D+i]*double(in.norm[i])*inv; }
        }
        down=project_reference(dw,xn,in.tokens); lo=down;
        for(double& v:lo) { v/=4; v*=logistic(v); }
        gate=project_reference(uw,lo,in.tokens);
        if(in.injection) inject=project_reference(iw,xn,in.tokens);
        for(int t=0;t<in.tokens;++t) for(size_t d=0;d<N;++d) {
            double sum=0; for(size_t c=0;c<H;++c) {size_t i=size_t(t)*D+c*N+d; sum+=xn[i]*logistic(gate[i]);}
            mixed[size_t(t)*N+d]=sum/4;
        }
    }
    std::vector<const std::vector<double>*> outputs(bool apply) const {
        return {apply ? &residual:nullptr,&rs,&xn,&down,&lo,&gate,&inject,&mixed};
    }
};
const char* stage_names[]={"pending_R","per_stream_rms","xn","down_pre_silu","lo_post_silu","up_raw_gate","injection","mixed"};
Numeric metric(const std::vector<double>& reference,const std::vector<float>& output) {
    if(reference.size()!=output.size()) throw std::runtime_error("metric size mismatch");
    Numeric n; double e2=0,r2=0,me=0,mr=0;
    for(size_t i=0;i<output.size();++i) {n.finite &= std::isfinite(output[i]); double e=double(output[i])-reference[i];e2+=e*e;r2+=reference[i]*reference[i];me=std::max(me,std::abs(e));mr=std::max(mr,std::abs(reference[i]));}
    n.nmse=e2/std::max(1e-30,r2);n.linf=me/std::max(1e-6,mr);return n;
}
#ifndef HC_FIXTURE_CPU_ONLY
struct CompositionBuffers {
    sycl::queue& q; const CompositionInput& input;
    DeviceBuffer norm,down,up,inject,R,bo,prior,Rout,rs,xn,pre,lo,gate,inj,mixed;
    strata::kernels::HcNativeArgs args;
    CompositionBuffers(sycl::queue& Q,const CompositionInput& in,const Fixture& dw,const Fixture& uw,const Fixture& iw) : q(Q),input(in),
        norm(q,in.norm.data(),D*4),down(q,dw.weights.data(),dw.weights.size()),up(q,uw.weights.data(),uw.weights.size()),inject(q,iw.weights.data(),iw.weights.size()),
        R(q,in.residual.data(),in.residual.size()*4),bo(q,in.bo.data(),in.bo.size()*4),prior(q,in.prior.data(),in.prior.size()*4),
        Rout(q,nullptr,in.tokens*D*4),rs(q,nullptr,in.tokens*H*4),xn(q,nullptr,in.tokens*D*4),pre(q,nullptr,in.tokens*L*4),lo(q,nullptr,in.tokens*L*4),
        gate(q,nullptr,in.tokens*D*4),inj(q,nullptr,in.tokens*H*4),mixed(q,nullptr,in.tokens*N*4) {
        args.w.norm=ptr(norm);args.w.norm_floats=D;args.w.down=down.data();args.w.down_bytes=dw.weights.size();args.w.up=up.data();args.w.up_bytes=uw.weights.size();
        if(in.injection) {args.w.inject=ptr(inject);args.w.inject_floats=iw.weights.size()/4;args.inject_out=ptr(inj);args.inject_out_floats=in.tokens*H;}
        args.tokens=in.tokens;args.R=ptr(R);args.R_floats=in.tokens*D;args.apply=in.apply;
        if(in.apply) {args.bo_prev=ptr(bo);args.bo_floats=in.tokens*N;args.inj_prev=ptr(prior);args.prev_floats=in.tokens*H;args.R_out=ptr(in.inplace ? R:Rout);args.R_out_floats=in.tokens*D;}
        args.rs=ptr(rs);args.rs_floats=in.tokens*H;args.xn=ptr(xn);args.xn_floats=in.tokens*D;args.lo=ptr(lo);args.lo_floats=in.tokens*L;
        args.gate=ptr(gate);args.gate_floats=in.tokens*D;args.mixed=ptr(mixed);args.mixed_floats=in.tokens*N;
    }
    static float* ptr(DeviceBuffer& b) {return reinterpret_cast<float*>(b.data());}
    void reset_R() {q.memcpy(R.data(),input.residual.data(),input.residual.size()*4).wait_and_throw();}
    void run(bool singles=false) {
        reset_R();
        if(!singles) {if(!strata::kernels::hc_native_read_f32(args,&q)) throw std::runtime_error("composition descriptor rejected");}
        else for(int t=0;t<input.tokens;++t) {
            auto a=args;a.tokens=1;a.R+=size_t(t)*D;a.R_floats=D;a.rs+=size_t(t)*H;a.rs_floats=H;a.xn+=size_t(t)*D;a.xn_floats=D;
            a.lo+=size_t(t)*L;a.lo_floats=L;a.gate+=size_t(t)*D;a.gate_floats=D;a.mixed+=size_t(t)*N;a.mixed_floats=N;
            if(input.apply) {a.bo_prev+=size_t(t)*N;a.bo_floats=N;a.inj_prev+=size_t(t)*H;a.prev_floats=H;a.R_out+=size_t(t)*D;a.R_out_floats=D;}
            if(input.injection) {a.inject_out+=size_t(t)*H;a.inject_out_floats=H;}
            if(!strata::kernels::hc_native_read_f32(a,&q)) throw std::runtime_error("single composition rejected");
        }
        // Expose the pre-SiLU down intermediate separately, using the composition's xn.
        if(!strata::kernels::hc_q8_0_project_f32(down.data(),args.w.down_bytes,ptr(xn),input.tokens*D,ptr(pre),input.tokens*L,D,L,input.tokens,&q)) throw std::runtime_error("pre-SiLU projection rejected");
        q.wait_and_throw();
    }
    std::vector<DeviceBuffer*> outputs() {return {input.apply ? (input.inplace ? &R:&Rout):nullptr,&rs,&xn,&pre,&lo,&gate,input.injection ? &inj:nullptr,&mixed};}
    std::vector<std::vector<uint8_t>> snapshots() {std::vector<std::vector<uint8_t>> v;for(auto* b:outputs()) v.push_back(b ? b->read():std::vector<uint8_t>{});return v;}
    bool guards_ok() {
        for(auto* b:{&norm,&down,&up,&inject,&R,&bo,&prior,&Rout,&rs,&xn,&pre,&lo,&gate,&inj,&mixed}) if(!b->valid(b->read())) return false;
        return true;
    }
    bool unchanged(const Fixture& dw,const Fixture& uw,const Fixture& iw) {
        return norm.valid(norm.read(),input.norm.data()) && down.valid(down.read(),dw.weights.data()) && up.valid(up.read(),uw.weights.data()) && inject.valid(inject.read(),iw.weights.data()) &&
            ((!input.apply || !input.inplace) ? R.valid(R.read(),input.residual.data()):true) && bo.valid(bo.read(),input.bo.data()) && prior.valid(prior.read(),input.prior.data());
    }
};
#endif
}
int main(int argc,char** argv) {
    try {
        std::ofstream receipt;std::ostream* log=&std::cout;
        if(argc>2) throw std::runtime_error("usage: composition_fixture [receipt.jsonl]");
        if(argc==2) {receipt.open(argv[1]);if(!receipt) throw std::runtime_error("cannot open receipt");log=&receipt;}
        *log<<std::setprecision(17)<<std::boolalpha;
        Fixture dw(D,L,1,0,true),uw(L,D,1,0,true),iw(D,H,1,0,false);
#ifndef HC_FIXTURE_CPU_ONLY
        sycl::queue q(sycl::gpu_selector_v,sycl::property::queue::in_order{});
        std::cerr<<"Selected GPU: "<<q.get_device().get_info<sycl::info::device::name>()<<'\n';
#endif
        int cases=0,failed=0,stages=0,negative_controls=0;
        for(int tokens:{1,2,4,8}) for(int pending=0;pending<3;++pending) for(bool injection:{false,true}) for(int profile:{0,1}) {
            CompositionInput input(tokens,pending!=0,injection,pending==2,profile);Oracle oracle(input,dw,uw,iw);
            auto refs=oracle.outputs(input.apply);std::vector<std::vector<float>> values(8);
            bool repeat=true,single=true,guards=true,unchanged=true;
#ifndef HC_FIXTURE_CPU_ONLY
            CompositionBuffers a(q,input,dw,uw,iw);
            a.run();auto first=a.snapshots();guards &= a.guards_ok();unchanged &= a.unchanged(dw,uw,iw);
            CompositionInput alternate=input;for(float& v:alternate.residual) v=v*0.5f+0.0001f;
            CompositionBuffers b(q,alternate,dw,uw,iw);b.run();
            a.run();repeat=first==a.snapshots();guards &= a.guards_ok() && b.guards_ok();unchanged &= a.unchanged(dw,uw,iw) && b.unchanged(dw,uw,iw);
            CompositionBuffers one(q,input,dw,uw,iw);one.run(true);single=first==one.snapshots();guards &= one.guards_ok();unchanged &= one.unchanged(dw,uw,iw);
            for(size_t s=0;s<8;++s) if(refs[s] && !refs[s]->empty()) {values[s].resize(refs[s]->size());std::memcpy(values[s].data(),first[s].data()+guard,values[s].size()*4);}
#else
            for(size_t s=0;s<8;++s) if(refs[s]) for(double v:*refs[s]) values[s].push_back(float(v));
#endif
            bool pass=repeat && single && guards && unchanged;
            for(size_t s=0;s<8;++s) if(refs[s] && !refs[s]->empty()) {
                Numeric n=metric(*refs[s],values[s]);bool stage_pass=n.finite && n.nmse<=1e-6 && n.linf<=1e-4;pass &= stage_pass;++stages;
                double mr=0;for(double v:*refs[s]) mr=std::max(mr,std::abs(v));auto corrupt=values[s];corrupt[0]=float((*refs[s])[0]+0.01*std::max(1e-6,mr));Numeric bad=metric(*refs[s],corrupt);
                if(bad.finite && bad.nmse<=1e-6 && bad.linf<=1e-4) throw std::runtime_error("composition numeric negative control accepted");
                ++negative_controls;
                *log<<"{\"kind\":\"stage\",\"case\":"<<cases<<",\"stage\":\""<<stage_names[s]<<"\",\"finite\":"<<n.finite<<",\"nmse\":";
                if(std::isfinite(n.nmse)) *log<<n.nmse;else *log<<"null";*log<<",\"normalized_linf\":";if(std::isfinite(n.linf)) *log<<n.linf;else *log<<"null";
                *log<<",\"pass\":"<<stage_pass<<"}\n";
            }
            if(!pass) ++failed;
            *log<<"{\"kind\":\"case\",\"case\":"<<cases<<",\"tokens\":"<<tokens<<",\"apply\":"<<input.apply<<",\"inplace\":"<<input.inplace<<",\"injection\":"<<injection<<",\"profile\":\""<<(profile ? "epsilon_sensitive":"mixed")<<"\",\"exact_repeat\":"<<repeat<<",\"exact_batch_vs_single\":"<<single<<",\"guards\":"<<guards<<",\"unchanged_weights_input\":"<<unchanged<<",\"pass\":"<<pass<<"}\n";log->flush();++cases;
        }
#ifdef HC_FIXTURE_CPU_ONLY
        const char* mode="CPU_REFERENCE_ONLY";
#else
        const char* mode="GPU_COMPOSITION_NUMERICAL";
#endif
        bool pass=cases==48 && failed==0 && stages==344 && negative_controls==344;
        *log<<"{\"kind\":\"summary\",\"mode\":\""<<mode<<"\",\"cases\":"<<cases<<",\"failed\":"<<failed<<",\"stages\":"<<stages<<",\"numeric_negative_controls\":"<<negative_controls<<",\"nmse_gate\":1e-6,\"normalized_linf_gate\":1e-4,\"pass\":"<<pass<<"}\n";
        return pass ? 0:1;
    }catch(const std::exception& e){std::cerr<<"Composition fixture failure: "<<e.what()<<'\n';return 2;}
}
