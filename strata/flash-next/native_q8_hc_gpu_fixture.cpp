// Standalone native Q8 HC qualification. Run GPU mode only under bin/gpu-run.
// -DHC_FIXTURE_CPU_ONLY validates fixture construction/reference without SYCL.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#ifndef HC_FIXTURE_CPU_ONLY
#include <sycl/sycl.hpp>
#include "strata/kernels/hc_native_projection.hpp"
#endif
namespace {
constexpr size_t guard = 128;
constexpr uint8_t sentinel = 0xa5;
const char* profiles[] = {"mixed", "negative_extreme", "positive_extreme", "cancellation", "impulse", "scale_edges"};
double scale_decode(uint16_t bits) {
    const int e = (bits >> 10) & 31, f = bits & 1023;
    if (e == 31) throw std::runtime_error("nonfinite fixture scale");
    return (bits & 0x8000 ? -1.0 : 1.0) * std::ldexp(double(e ? 1024 + f : f), e ? e - 25 : -24);
}
float bf16_round(float f) {
    uint32_t u; std::memcpy(&u, &f, 4);
    u = (u + 0x7fff + ((u >> 16) & 1)) & 0xffff0000;
    std::memcpy(&f, &u, 4); return f;
}
struct Fixture {
    int k, m, tokens, profile; bool q8;
    std::vector<uint8_t> weights;
    std::vector<float> input;
    std::vector<double> reference;
    double bf16_difference = 0;
    Fixture(int K, int M, int T, int P, bool Q) : k(K), m(M), tokens(T), profile(P), q8(Q),
        weights(Q ? size_t(M)*(K/32)*34 : size_t(M)*K*4), input(size_t(T)*K), reference(size_t(T)*M) {
        const uint16_t scales[] = {0,0x8000,0x0400,0x8400,1,0x8001,0x0200,0x8200,0x2400,0xa400};
        if (q8) {
            for (int r=0; r<m; ++r) for (int b=0; b<k/32; ++b) {
                size_t o=(size_t(r)*(k/32)+b)*34;
                uint16_t s = profile==5 ? scales[(r+b)%10] : (profile==3 ? 0x2400 : uint16_t(((r+b)%3==0 ? 0x8000:0) | (0x2400 + ((r*17+b*13)%1024))));
                weights[o]=s&255; weights[o+1]=s>>8;
                for (int l=0;l<32;++l) {
                    int c = profile==1 ? -128 : profile==2 ? 127 : profile==3 ? (l%2 ? -127 : 127) : int((r*29+b*19+l*37)%256)-128;
                    weights[o+2+l]=uint8_t(c & 255);
                }
            }
        } else {
            for (int r=0;r<m;++r) for (int c=0;c<k;++c) {
                float w = profile==1 ? -1.001f : profile==2 ? 1.001f : profile==3 ? (c%2 ? -1.001f:1.001f) :
                    (float((r*13+c*17)%127)-63.0f)*0.015631f + 0.000123f;
                if (profile==5) w = (c%2 ? -1.0f:1.0f)*std::ldexp(1.001f, -(c%12));
                std::memcpy(weights.data()+(size_t(r)*k+c)*4,&w,4);
            }
        }
        for (int t=0;t<tokens;++t) for (int c=0;c<k;++c) {
            input[size_t(t)*k+c] = profile==4 ? (c==(t*37)%k ? 1.001f:0.0f) :
                profile==3 ? float((c/2+t)%19+1)*0.03125f : float((c*31+t*43)%251-125)*0.007813f;
        }
        for (int t=0;t<tokens;++t) for (int r=0;r<m;++r) {
            double sum=0, control=0;
            for (int c=0;c<k;++c) {
                double w;
                if (q8) {
                    size_t o=(size_t(r)*(k/32)+c/32)*34;
                    uint16_t bits=uint16_t(weights[o]) | (uint16_t(weights[o+1])<<8);
                    int raw=weights[o+2+c%32];
                    w=scale_decode(bits)*(raw<128 ? raw:raw-256);
                } else {
                    float f; std::memcpy(&f,weights.data()+(size_t(r)*k+c)*4,4); w=f;
                    control += double(bf16_round(f))*double(input[size_t(t)*k+c]);
                }
                sum += w*double(input[size_t(t)*k+c]);
            }
            reference[size_t(t)*m+r]=sum;
            if (!q8) bf16_difference=std::max(bf16_difference,std::abs(sum-control));
        }
        for (double v : reference) {
            if (!std::isfinite(v)) throw std::runtime_error("nonfinite CPU reference");
            if (profile==3 && v!=0.0) throw std::runtime_error("cancellation fixture did not cancel exactly");
        }
        if (!q8 && profile!=3 && !(bf16_difference>0)) throw std::runtime_error("F32 fixture has no BF16-sensitive reference");
    }
};
struct Numeric { double nmse=0, linf=0; bool finite=true; };
Numeric measure(const Fixture& f, const std::vector<float>& out) {
    Numeric n; double error2=0, ref2=0, maxref=0, maxerr=0;
    for (size_t i=0;i<out.size();++i) {
        n.finite = n.finite && std::isfinite(out[i]);
        double e=double(out[i])-f.reference[i];
        error2+=e*e; ref2+=f.reference[i]*f.reference[i];
        maxref=std::max(maxref,std::abs(f.reference[i])); maxerr=std::max(maxerr,std::abs(e));
    }
    n.nmse=error2/std::max(1e-30,ref2); n.linf=maxerr/std::max(1e-6,maxref); return n;
}
#ifndef HC_FIXTURE_CPU_ONLY
struct DeviceBuffer {
    sycl::queue& q; uint8_t* base; size_t bytes;
    DeviceBuffer(sycl::queue& Q, const void* data, size_t N) : q(Q), bytes(N) {
        base=static_cast<uint8_t*>(sycl::malloc_device(N+2*guard,q));
        if (!base) throw std::runtime_error("device allocation failed");
        std::vector<uint8_t> image(N+2*guard,sentinel);
        if (data) std::memcpy(image.data()+guard,data,N);
        q.memcpy(base,image.data(),image.size()).wait_and_throw();
    }
    ~DeviceBuffer() { sycl::free(base,q); }
    uint8_t* data() { return base+guard; }
    std::vector<uint8_t> read() {
        std::vector<uint8_t> image(bytes+2*guard); q.memcpy(image.data(),base,image.size()).wait_and_throw(); return image;
    }
    bool valid(const std::vector<uint8_t>& image, const void* original=nullptr) {
        bool ok=std::all_of(image.begin(),image.begin()+guard,[](uint8_t b){return b==sentinel;}) &&
            std::all_of(image.end()-guard,image.end(),[](uint8_t b){return b==sentinel;});
        return ok && (!original || std::memcmp(image.data()+guard,original,bytes)==0);
    }
    DeviceBuffer(const DeviceBuffer&)=delete;
};
bool launch(const Fixture& f, DeviceBuffer& w, DeviceBuffer& x, float* y, int tokens, sycl::queue& q, const float* input=nullptr) {
    if (!input) input=reinterpret_cast<float*>(x.data());
    if (f.q8) return strata::kernels::hc_q8_0_project_f32(w.data(),f.weights.size(),input,size_t(f.k)*tokens,y,size_t(f.m)*tokens,f.k,f.m,tokens,&q);
    return strata::kernels::hc_f32_project_f32(reinterpret_cast<float*>(w.data()),f.weights.size()/4,input,size_t(f.k)*tokens,y,size_t(f.m)*tokens,f.k,f.m,tokens,&q);
}
int reject_tests(sycl::queue& q) {
    Fixture f(32,1,1,0,true); DeviceBuffer w(q,f.weights.data(),f.weights.size()), x(q,f.input.data(),128), y(q,nullptr,4);
    auto before=y.read(); auto* xp=reinterpret_cast<float*>(x.data()); auto* yp=reinterpret_cast<float*>(y.data());
    int count=0;
    auto reject=[&](const uint8_t* wp,size_t wb,const float* ip,size_t ni,float* op,size_t no,int k,int m,int t,void* queue) {
        if (strata::kernels::hc_q8_0_project_f32(wp,wb,ip,ni,op,no,k,m,t,queue)) throw std::runtime_error("invalid Q8 descriptor accepted"); ++count;
    };
    reject(nullptr,34,xp,32,yp,1,32,1,1,&q); reject(w.data(),34,nullptr,32,yp,1,32,1,1,&q);
    reject(w.data(),34,xp,32,nullptr,1,32,1,1,&q); reject(w.data(),33,xp,32,yp,1,32,1,1,&q);
    reject(w.data(),34,xp,31,yp,1,32,1,1,&q); reject(w.data(),34,xp,32,yp,0,32,1,1,&q);
    reject(w.data(),34,xp,32,yp,1,33,1,1,&q); reject(w.data(),34,xp,32,yp,1,0,1,1,&q);
    reject(w.data(),34,xp,32,yp,1,32,0,1,&q); reject(w.data(),34,xp,32,yp,1,32,1,0,&q);
    reject(w.data(),34,xp,288,yp,9,32,1,9,&q);
    sycl::queue unordered(q.get_context(),q.get_device());
    reject(w.data(),34,xp,32,yp,1,32,1,1,&unordered);
    auto* wf=reinterpret_cast<float*>(w.data());
    auto freject=[&](const float* wp,size_t nw,const float* ip,size_t ni,float* op,size_t no,int k,int m,int t,void* queue) {
        if (strata::kernels::hc_f32_project_f32(wp,nw,ip,ni,op,no,k,m,t,queue)) throw std::runtime_error("invalid F32 descriptor accepted"); ++count;
    };
    freject(nullptr,32,xp,32,yp,1,32,1,1,&q); freject(wf,32,nullptr,32,yp,1,32,1,1,&q);
    freject(wf,32,xp,32,nullptr,1,32,1,1,&q); freject(wf,31,xp,32,yp,1,32,1,1,&q);
    freject(wf,32,xp,31,yp,1,32,1,1,&q); freject(wf,32,xp,32,yp,0,32,1,1,&q);
    freject(wf,32,xp,32,yp,1,33,1,1,&q); freject(wf,32,xp,32,yp,1,32,1,0,&q);
    freject(wf,32,xp,288,yp,9,32,1,9,&q); freject(wf,32,xp,32,yp,1,32,1,1,&unordered);
    q.wait_and_throw(); unordered.wait_and_throw();
    if (y.read()!=before || !w.valid(w.read(),f.weights.data()) || !x.valid(x.read(),f.input.data())) throw std::runtime_error("rejected call changed storage");
    return count;
}
#endif
}
int main(int argc,char** argv) {
    try {
        std::ofstream receipt; std::ostream* log=&std::cout;
        if (argc>2) throw std::runtime_error("usage: fixture [receipt.jsonl]");
        if (argc==2) { receipt.open(argv[1]); if (!receipt) throw std::runtime_error("cannot open receipt"); log=&receipt; }
        *log << std::setprecision(17) << std::boolalpha;
        int cases=0, failed=0, bf_sensitive=0, rejected=0;
#ifndef HC_FIXTURE_CPU_ONLY
        sycl::queue q(sycl::gpu_selector_v,sycl::property::queue::in_order{});
        // Device name is printed separately on stderr; JSON fields stay ASCII and bounded.
        std::cerr << "Selected GPU: " << q.get_device().get_info<sycl::info::device::name>() << '\n';
        rejected=reject_tests(q);
        DeviceBuffer guard_control(q,nullptr,4);
        auto corrupt=guard_control.read(); corrupt.front()=0;
        if (guard_control.valid(corrupt)) throw std::runtime_error("guard negative control accepted corruption");
#endif
        for (int tokens : {1,2,4,8}) for (int role=0;role<3;++role) for (int profile=0;profile<6;++profile) {
            Fixture f(role==1 ? 320:10240,role==0 ? 320:role==1 ? 10240:4,tokens,profile,role!=2);
            bool repeat=true, single=true, guards=true, unchanged=true; std::vector<float> out(f.reference.size());
#ifdef HC_FIXTURE_CPU_ONLY
            for (size_t i=0;i<out.size();++i) out[i]=float(f.reference[i]);
#else
            DeviceBuffer w(q,f.weights.data(),f.weights.size()), x(q,f.input.data(),f.input.size()*4), y(q,nullptr,out.size()*4);
            // Independent buffer set B changes input, then A is replayed: expose hidden shared state.
            auto other=f.input; for (float& v:other) v=v*0.5f+0.125f;
            DeviceBuffer wx(q,f.weights.data(),f.weights.size()), xx(q,other.data(),other.size()*4), yy(q,nullptr,out.size()*4);
            if (!launch(f,w,x,reinterpret_cast<float*>(y.data()),tokens,q)) throw std::runtime_error("valid descriptor rejected");
            q.wait_and_throw(); auto first=y.read(); std::memcpy(out.data(),first.data()+guard,out.size()*4);
            guards &= y.valid(first);
            if (!launch(f,wx,xx,reinterpret_cast<float*>(yy.data()),tokens,q)) throw std::runtime_error("independent descriptor rejected");
            if (!launch(f,w,x,reinterpret_cast<float*>(y.data()),tokens,q)) throw std::runtime_error("repeat rejected");
            q.wait_and_throw(); auto again=y.read(); repeat=first==again; guards &= y.valid(again) && yy.valid(yy.read());
            DeviceBuffer singles(q,nullptr,out.size()*4);
            for (int t=0;t<tokens;++t) if (!launch(f,w,x,reinterpret_cast<float*>(singles.data())+size_t(t)*f.m,1,q,reinterpret_cast<float*>(x.data())+size_t(t)*f.k)) throw std::runtime_error("single descriptor rejected");
            q.wait_and_throw(); auto combined=singles.read(); single=first==combined; guards &= singles.valid(combined);
            unchanged = w.valid(w.read(),f.weights.data()) && x.valid(x.read(),f.input.data()) && wx.valid(wx.read(),f.weights.data()) && xx.valid(xx.read(),other.data());
#endif
            // Deliberately perturb a host output by more than the preregistered
            // normalized maximum-error gate. This must fail every fixture oracle.
            auto corrupt_output=out;
            double maxref=0;
            for (double v:f.reference) maxref=std::max(maxref,std::abs(v));
            corrupt_output[0]=float(f.reference[0]+0.01*std::max(1e-6,maxref));
            Numeric bad=measure(f,corrupt_output);
            if (bad.finite && bad.nmse<=1e-6 && bad.linf<=1e-4) throw std::runtime_error("numeric negative control accepted perturbation");
            Numeric n=measure(f,out);
            bool pass=n.finite && n.nmse<=1e-6 && n.linf<=1e-4 && repeat && single && guards && unchanged;
            if (!f.q8 && profile!=3 && f.bf16_difference>0) ++bf_sensitive;
            if (!pass) ++failed;
            ++cases;
            *log << "{\"kind\":\"case\",\"role\":\"" << (role==0 ? "down":role==1 ? "up":"injection") << "\",\"profile\":\"" << profiles[profile]
                 << "\",\"k\":" << f.k << ",\"m\":" << f.m << ",\"tokens\":" << tokens;
            // Nonfinite output fails explicitly; avoid emitting invalid JSON NaN/Inf.
            *log << ",\"finite\":" << n.finite << ",\"nmse\":";
            if(std::isfinite(n.nmse)) *log << n.nmse; else *log << "null";
            *log << ",\"normalized_linf\":"; if(std::isfinite(n.linf)) *log << n.linf; else *log << "null";
            *log << ",\"exact_repeat\":" << repeat << ",\"exact_batch_vs_single\":" << single << ",\"guards\":" << guards << ",\"unchanged_weights_input\":" << unchanged
                 << ",\"bf16_reference_max_difference\":" << f.bf16_difference << ",\"pass\":" << pass << "}\n"; log->flush();
        }
        bool pass=failed==0 && cases==72 && bf_sensitive==20;
#ifdef HC_FIXTURE_CPU_ONLY
        const char* mode="CPU_REFERENCE_ONLY";
#else
        const char* mode="GPU_NUMERICAL";
        pass=pass && rejected==22;
#endif
        *log << "{\"kind\":\"summary\",\"mode\":\"" << mode << "\",\"shape_cases\":12,\"profiles_per_shape\":6,\"cases\":" << cases << ",\"failed\":" << failed
             << ",\"bf16_sensitive_cases\":" << bf_sensitive << ",\"numeric_negative_controls\":72,\"guard_negative_control\":" << (rejected==22) << ",\"rejection_cases\":" << rejected << ",\"nmse_gate\":1e-6,\"normalized_linf_gate\":1e-4,\"pass\":" << pass << "}\n";
        return pass ? 0:1;
    } catch (const std::exception& e) { std::cerr << "Fixture failure: " << e.what() << '\n'; return 2; }
}
