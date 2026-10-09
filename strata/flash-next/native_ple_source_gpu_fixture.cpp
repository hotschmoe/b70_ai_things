// Production-shape original-source PLE qualification; GPU runs need bin/gpu-run.
#define main hc_projection_fixture_unused_main
#include "native_q8_hc_gpu_fixture.cpp"
#undef main
#include "strata/kernels/native_ple_postops.hpp"
#include <memory>
namespace {
constexpr size_t PN=2560,PH=4,PD=10240,HH=9;
const char* ple_stages[]={"key_raw","value_raw","key_norm","query_norm","gate","gated","norm_conv","conv_silu","result","history_snapshots"};
float half_round(float f) {
    uint32_t x;std::memcpy(&x,&f,4);uint16_t sign=(x>>16)&0x8000;int e=int((x>>23)&255)-112;uint32_t m=x&0x7fffff,h;
    if(e<=0) {if(e < -10) return float(scale_decode(sign));m|=0x800000;unsigned shift=unsigned(14-e);h=m>>shift;uint32_t rem=m&((1u<<shift)-1),mid=1u<<(shift-1);if(rem>mid || (rem==mid && (h&1))) ++h;}
    else {if(e>=31) throw std::runtime_error("nonfinite half control");h=(uint32_t(e)<<10)|(m>>13);uint32_t rem=m&8191;if(rem>4096 || (rem==4096 && (h&1))) ++h;}
    return float(scale_decode(uint16_t(sign|h)));
}
double qweight(const Fixture& w,size_t r,size_t c) {size_t off=(r*(w.k/32)+c/32)*34;uint16_t bits=uint16_t(w.weights[off])|(uint16_t(w.weights[off+1])<<8);int raw=w.weights[off+2+c%32];return scale_decode(bits)*(raw<128 ? raw:raw-256);}
std::vector<double> ple_project(const Fixture& w,const std::vector<float>& x,int rows) {
    std::vector<double> y(size_t(rows)*w.m);
    for(int t=0;t<rows;++t) for(int r=0;r<w.m;++r) {double sum=0;for(int c=0;c<w.k;++c) sum+=qweight(w,r,c)*double(x[size_t(t)*w.k+c]);y[size_t(t)*w.m+r]=sum;}
    return y;
}
void norm_reference(const double* input,const std::vector<float>& gamma,double* out) {
    for(size_t c=0;c<PH;++c) {double sum=0;for(size_t d=0;d<PN;++d) {double v=input[c*PN+d];sum+=v*v;}double rs=1/std::sqrt(sum/PN+double(1e-6f));for(size_t d=0;d<PN;++d) out[c*PN+d]=rs*input[c*PN+d]*double(gamma[c*PN+d]);}
}
struct PleInput {
    int rows;std::vector<float> emb,hidden,history,nk,nq,nc,conv;
    PleInput(int t):rows(t),emb(size_t(t)*PN),hidden(size_t(t)*PD),history(HH*PD),nk(PD),nq(PD),nc(PD),conv(4*PD) {
        for(size_t i=0;i<emb.size();++i) emb[i]=float(int((i*31+i/PN*19)%251)-125)*0.007813f+0.000123f;
        for(size_t i=0;i<hidden.size();++i) hidden[i]=float(int((i*17+i/PD*7)%127)-63)*0.031251f;
        for(size_t i=0;i<history.size();++i) history[i]=float(int((i*13)%71)-35)*0.015631f;
        for(size_t i=0;i<PD;++i) {nk[i]=0.800123f+float(i%7)*0.019531f;nq[i]=0.900123f+float(i%13)*0.015631f;nc[i]=1.000123f+float(i%11)*0.009763f;}
        for(size_t i=0;i<conv.size();++i) conv[i]=float(int((i*7)%23)-11)*0.00100013f;
    }
};
struct PleOracle {
    std::vector<std::vector<double>> out;double bf16_value_difference=0,f16_activation_difference=0,q8_activation_difference=0,f16_conv_difference=0;
    PleOracle(PleInput& input,const Fixture& key,const Fixture& val):out(10) {
        out[0]=ple_project(key,input.emb,input.rows);out[1]=ple_project(val,input.emb,input.rows);
        // Deliberately exercise aligned, anti-aligned and mixed key/query gates.
        for(int t=0;t<input.rows;++t) if(t%3!=2) for(size_t d=0;d<PD;++d) input.hidden[size_t(t)*PD+d]=float(out[0][size_t(t)*PD+d])*(t%3 ? -1.0f:1.0f);
        for(int s:{2,3,5,6,7,8}) out[s].resize(size_t(input.rows)*PD);
        out[4].resize(size_t(input.rows)*PH);out[9].resize(size_t(input.rows)*HH*PD);
        std::vector<double> history(input.history.begin(),input.history.end());
        for(int t=0;t<input.rows;++t) {
            size_t off=size_t(t)*PD;norm_reference(out[0].data()+off,input.nk,out[2].data()+off);
            std::vector<double> hidden(input.hidden.begin()+off,input.hidden.begin()+off+PD);norm_reference(hidden.data(),input.nq,out[3].data()+off);
            for(size_t c=0;c<PH;++c) {double sum=0;for(size_t d=0;d<PN;++d) sum+=out[2][off+c*PN+d]*out[3][off+c*PN+d];double score=sum/std::sqrt(double(PN));double sign=(score>0)-(score<0);double g=1/(1+std::exp(-sign*std::sqrt(std::max(std::abs(score),1e-6))));out[4][size_t(t)*PH+c]=g;for(size_t d=0;d<PN;++d) out[5][off+c*PN+d]=out[1][size_t(t)*PN+d]*g;}
            norm_reference(out[5].data()+off,input.nc,out[6].data()+off);
            for(size_t c=0;c<PD;++c) {double sum=0,control=0;for(size_t k=0;k<4;++k) {double x=k==3 ? out[6][off+c]:history[c*HH+3*k];sum+=double(input.conv[c*4+k])*x;control+=double(half_round(input.conv[c*4+k]))*x;}double activation=sum/(1+std::exp(-sum));out[7][off+c]=activation;out[8][off+c]=hidden[c]+out[5][off+c]+activation;f16_conv_difference=std::max(f16_conv_difference,std::abs(sum-control));}
            for(size_t c=0;c<PD;++c) {for(size_t h=0;h<HH-1;++h) history[c*HH+h]=history[c*HH+h+1];history[c*HH+HH-1]=out[6][off+c];}
            std::copy(history.begin(),history.end(),out[9].begin()+size_t(t)*HH*PD);
        }
        double value_control=0,key_f16=0,key_q8=0;
        std::vector<float> quant(PN);
        for(size_t b=0;b<PN/32;++b) {float amax=0;for(size_t l=0;l<32;++l) amax=std::max(amax,std::abs(input.emb[b*32+l]));float scale=amax/127;for(size_t l=0;l<32;++l) quant[b*32+l]=scale ? half_round(scale)*float(std::round(input.emb[b*32+l]/scale)):0;}
        for(size_t c=0;c<PN;++c) {value_control+=double(bf16_round(float(qweight(val,0,c))))*double(input.emb[c]);key_f16+=qweight(key,0,c)*double(half_round(input.emb[c]));key_q8+=qweight(key,0,c)*double(quant[c]);}
        bf16_value_difference=std::abs(out[1][0]-value_control);f16_activation_difference=std::abs(out[0][0]-key_f16);q8_activation_difference=std::abs(out[0][0]-key_q8);
        if(!(bf16_value_difference>0 && f16_activation_difference>0 && q8_activation_difference>0 && f16_conv_difference>0)) throw std::runtime_error("PLE lossy conversion control had no effect");
    }
};
Numeric ple_metric(const std::vector<double>& ref,const std::vector<float>& values) {
    if(ref.size()!=values.size()) throw std::runtime_error("PLE metric size mismatch");
    Numeric n;double e2=0,r2=0,me=0,mr=0;
    for(size_t i=0;i<values.size();++i) {n.finite &= std::isfinite(values[i]);double e=double(values[i])-ref[i];e2+=e*e;r2+=ref[i]*ref[i];me=std::max(me,std::abs(e));mr=std::max(mr,std::abs(ref[i]));}n.nmse=e2/std::max(1e-30,r2);n.linf=me/std::max(1e-6,mr);return n;
}
#ifndef HC_FIXTURE_CPU_ONLY
class FixturePleHistory;
struct PleBuffers {
    sycl::queue& q;const PleInput& input;DeviceBuffer kw,vw,emb,hidden,hist,nk,nq,nc,cw;std::vector<std::unique_ptr<DeviceBuffer>> outputs;
    PleBuffers(sycl::queue& Q,const PleInput& in,const Fixture& key,const Fixture& val):q(Q),input(in),kw(q,key.weights.data(),key.weights.size()),vw(q,val.weights.data(),val.weights.size()),emb(q,in.emb.data(),in.emb.size()*4),hidden(q,in.hidden.data(),in.hidden.size()*4),hist(q,in.history.data(),in.history.size()*4),nk(q,in.nk.data(),PD*4),nq(q,in.nq.data(),PD*4),nc(q,in.nc.data(),PD*4),cw(q,in.conv.data(),in.conv.size()*4) {
        for(size_t s=0;s<10;++s) {size_t n=size_t(in.rows)*(s==1 ? PN:s==4 ? PH:s==9 ? HH*PD:PD);outputs.emplace_back(std::make_unique<DeviceBuffer>(q,nullptr,n*4));}
    }
    static float* ptr(DeviceBuffer& b){return reinterpret_cast<float*>(b.data());}
    float* out(size_t s){return ptr(*outputs[s]);}
    void run(int chunk) {
        q.memcpy(hist.data(),input.history.data(),input.history.size()*4).wait_and_throw();
        for(int t=0;t<input.rows;t+=chunk) {int rows=std::min(chunk,input.rows-t);const float* x=ptr(emb)+size_t(t)*PN;
            if(!strata::kernels::hc_q8_0_project_f32(kw.data(),kw.bytes,x,size_t(rows)*PN,out(0)+size_t(t)*PD,size_t(rows)*PD,PN,PD,rows,&q) || !strata::kernels::hc_q8_0_project_f32(vw.data(),vw.bytes,x,size_t(rows)*PN,out(1)+size_t(t)*PN,size_t(rows)*PN,PN,PN,rows,&q)) throw std::runtime_error("PLE projection rejected");
        }
        strata::kernels::PleWeights w;w.source_exact=true;w.key_source_q8=kw.data();w.key_source_bytes=kw.bytes;w.value_source_q8=vw.data();w.value_source_bytes=vw.bytes;w.conv_source_f32=ptr(cw);w.conv_source_floats=4*PD;w.norm_key=ptr(nk);w.norm_query=ptr(nq);w.norm_conv=ptr(nc);
        for(int t=0;t<input.rows;++t) {size_t off=size_t(t)*PD;strata::kernels::NativePlePostopsBuffers b{out(2)+off,out(3)+off,out(4)+size_t(t)*PH,out(5)+off,out(6)+off,out(7)+off,out(8)+off};
            strata::kernels::native_ple_postops(out(0)+off,ptr(hidden)+off,out(1)+size_t(t)*PN,ptr(hist),w,b,&q);
            // Fixture-owned caller protocol: one channel owner shifts its9 rows.
            // This tests postops against histories; it does not qualify engine capture.
            float* h=ptr(hist);float* normalized=out(6)+off;
            q.parallel_for<FixturePleHistory>(sycl::range<1>(PD),[=](sycl::id<1> index){size_t c=index[0];for(size_t r=0;r<HH-1;++r) h[c*HH+r]=h[c*HH+r+1];h[c*HH+HH-1]=normalized[c];});
            q.memcpy(out(9)+size_t(t)*HH*PD,h,HH*PD*4);
        }
        q.wait_and_throw();
    }
    std::vector<std::vector<uint8_t>> snapshots(){std::vector<std::vector<uint8_t>> v;for(auto& b:outputs) v.push_back(b->read());return v;}
    bool guards_ok(){for(auto* b:{&kw,&vw,&emb,&hidden,&hist,&nk,&nq,&nc,&cw}) if(!b->valid(b->read())) return false;for(auto& b:outputs) if(!b->valid(b->read())) return false;
        return true;}
    bool immutable(const Fixture& key,const Fixture& val){return kw.valid(kw.read(),key.weights.data()) && vw.valid(vw.read(),val.weights.data()) && emb.valid(emb.read(),input.emb.data()) && hidden.valid(hidden.read(),input.hidden.data()) && nk.valid(nk.read(),input.nk.data()) && nq.valid(nq.read(),input.nq.data()) && nc.valid(nc.read(),input.nc.data()) && cw.valid(cw.read(),input.conv.data());}
};
#endif
}
int main(int argc,char** argv) {
    try {
        std::ofstream receipt;std::ostream* log=&std::cout;if(argc>2) throw std::runtime_error("usage: PLE fixture [receipt.jsonl]");if(argc==2) {receipt.open(argv[1]);if(!receipt) throw std::runtime_error("cannot open receipt");log=&receipt;}*log<<std::setprecision(17)<<std::boolalpha;
        Fixture key(PN,PD,1,0,true),val(PN,PN,1,0,true);int cases=0,failed=0,stages=0,history_rows=0,negatives=0;bool guard_negative=false;
#ifndef HC_FIXTURE_CPU_ONLY
        sycl::queue q(sycl::gpu_selector_v,sycl::property::queue::in_order{});std::cerr<<"Selected GPU: "<<q.get_device().get_info<sycl::info::device::name>()<<'\n';DeviceBuffer probe(q,nullptr,4);auto bad_guard=probe.read();bad_guard.front()=0;guard_negative=!probe.valid(bad_guard);if(!guard_negative) throw std::runtime_error("PLE guard negative control accepted");
#endif
        for(int rows:{1,2,4,8,9,16,17}) {
            PleInput input(rows);PleOracle oracle(input,key,val);std::vector<std::vector<float>> values(10);bool repeat=true,single=true,guards=true,unchanged=true;
#ifndef HC_FIXTURE_CPU_ONLY
            PleBuffers a(q,input,key,val);a.run(8);auto first=a.snapshots();guards &= a.guards_ok();unchanged &= a.immutable(key,val);
            PleInput alternate=input;for(float& v:alternate.hidden) v=v*0.5f+0.000123f;{PleBuffers b(q,alternate,key,val);b.run(8);guards &= b.guards_ok();unchanged &= b.immutable(key,val);}
            a.run(8);repeat=first==a.snapshots();guards &= a.guards_ok();unchanged &= a.immutable(key,val);
            {PleBuffers singles(q,input,key,val);singles.run(1);single=first==singles.snapshots();guards &= singles.guards_ok();unchanged &= singles.immutable(key,val);}
            for(size_t s=0;s<10;++s) {values[s].resize(oracle.out[s].size());std::memcpy(values[s].data(),first[s].data()+guard,values[s].size()*4);}
#else
            for(size_t s=0;s<10;++s) for(double v:oracle.out[s]) values[s].push_back(float(v));
#endif
            bool pass=repeat && single && guards && unchanged;
            for(size_t s=0;s<10;++s) {Numeric n=ple_metric(oracle.out[s],values[s]);bool stage_pass=n.finite && n.nmse<=1e-6 && n.linf<=1e-4;pass &= stage_pass;++stages;
                double mr=0;for(double v:oracle.out[s]) mr=std::max(mr,std::abs(v));auto bad=values[s];bad[0]=float(oracle.out[s][0]+0.01*std::max(1e-6,mr));Numeric control=ple_metric(oracle.out[s],bad);if(control.finite && control.nmse<=1e-6 && control.linf<=1e-4) throw std::runtime_error("PLE numeric negative control accepted");++negatives;
                *log<<"{\"kind\":\"stage\",\"rows\":"<<rows<<",\"stage\":\""<<ple_stages[s]<<"\",\"finite\":"<<n.finite<<",\"nmse\":";if(std::isfinite(n.nmse)) *log<<n.nmse;else *log<<"null";*log<<",\"normalized_linf\":";if(std::isfinite(n.linf)) *log<<n.linf;else *log<<"null";*log<<",\"pass\":"<<stage_pass<<"}\n";
            }
            if(!pass) ++failed;
            history_rows+=rows;
            *log<<"{\"kind\":\"case\",\"rows\":"<<rows<<",\"exact_repeat\":"<<repeat<<",\"exact_chunk_vs_single\":"<<single<<",\"guards\":"<<guards<<",\"unchanged_weights_input\":"<<unchanged<<",\"bf16_value_reference_difference\":"<<oracle.bf16_value_difference<<",\"f16_activation_reference_difference\":"<<oracle.f16_activation_difference<<",\"q8_activation_reference_difference\":"<<oracle.q8_activation_difference<<",\"f16_conv_reference_difference\":"<<oracle.f16_conv_difference<<",\"pass\":"<<pass<<"}\n";log->flush();++cases;
        }
#ifdef HC_FIXTURE_CPU_ONLY
        const char* mode="CPU_REFERENCE_ONLY";bool pass=cases==7 && failed==0 && stages==70 && history_rows==57 && negatives==70;
#else
        const char* mode="GPU_PLE_SOURCE_NUMERICAL";bool pass=cases==7 && failed==0 && stages==70 && history_rows==57 && negatives==70 && guard_negative;
#endif
        *log<<"{\"kind\":\"summary\",\"mode\":\""<<mode<<"\",\"cases\":"<<cases<<",\"failed\":"<<failed<<",\"stages\":"<<stages<<",\"history_rows\":"<<history_rows<<",\"numeric_negative_controls\":"<<negatives<<",\"guard_negative_control\":"<<guard_negative<<",\"nmse_gate\":1e-6,\"normalized_linf_gate\":1e-4,\"pass\":"<<pass<<"}\n";return pass ? 0:1;
    }catch(const std::exception& e){std::cerr<<"PLE fixture failure: "<<e.what()<<'\n';return 2;}
}
