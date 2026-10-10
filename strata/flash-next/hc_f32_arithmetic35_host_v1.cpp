// Separate source-indexed host prototype; never an original/full-model oracle.
// Root compiles with -fno-fast-math -ffp-contract=off -frounding-math.
#include <array>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#if defined(__SSE__)
#include <xmmintrin.h>
#endif
namespace fs=std::filesystem;
std::vector<uint8_t> bytes(const char* path,size_t extent) {
    if(fs::is_symlink(path) || fs::file_size(path)!=extent)throw std::runtime_error("input extent/symlink differs");
    std::ifstream in(path,std::ios::binary);std::vector<uint8_t> out(extent);
    in.read(reinterpret_cast<char*>(out.data()),std::streamsize(extent));
    if(!in || in.peek()!=std::char_traits<char>::eof())throw std::runtime_error("input changed/short");
    return out;
}
std::vector<float> floats(const char* path,size_t n) {
    auto raw=bytes(path,n*4);std::vector<float> out(n);std::memcpy(out.data(),raw.data(),raw.size());
    for(float x:out)if(!std::isfinite(x))throw std::runtime_error("nonfinite input");return out;
}
float half(uint16_t bits) {
    const unsigned e=(bits>>10)&31,f=bits&1023;
    if(e==31)throw std::runtime_error("nonfinite half scale");
    float x=e?std::ldexp(float(1024+f),int(e)-25):std::ldexp(float(f),-24);
    return bits&0x8000?-x:x;
}
float dot32(const std::vector<float>& w,const std::vector<float>& x) {
    std::array<float,32> lanes{};
    for(size_t lane=0;lane<32;++lane)
        for(size_t column=lane;column<x.size();column+=32)
            lanes[lane]=::fmaf(w[column],x[column],lanes[lane]);
    for(unsigned mask=16;mask;mask>>=1) {
        auto prior=lanes;
        for(size_t lane=0;lane<32;++lane)lanes[lane]=prior[lane]+prior[lane^mask];
    }
    return lanes[0];
}
int main(int argc,char** argv) try {
    static_assert(sizeof(float)==4 && std::numeric_limits<float>::is_iec559);
    const uint32_t endian=1;if(*reinterpret_cast<const uint8_t*>(&endian)!=1)throw std::runtime_error("LE host required");
    if(argc!=8)throw std::runtime_error("usage: host OP N A B C-or-dash OUTPUT SOURCE_TAG");
    const std::string op=argv[1],tag=argv[7];size_t used=0,n=std::stoull(argv[2],&used);
    if(used!=std::strlen(argv[2]) || n<1 || n>10240 || tag!="source35_hc_f32_v1")throw std::runtime_error("bounded source/input geometry differs");
    if(std::fesetround(FE_TONEAREST)!=0 || std::fegetround()!=FE_TONEAREST)throw std::runtime_error("RNE host mode unavailable");
#if defined(__SSE__)
    _mm_setcsr(_mm_getcsr() & ~(unsigned(1)<<15) & ~(unsigned(1)<<6)); // FTZ/DAZ OFF.
    if(_mm_getcsr() & ((unsigned(1)<<15)|(unsigned(1)<<6)))throw std::runtime_error("gradual underflow host mode unavailable");
#else
    throw std::runtime_error("this prototype requires x86 SSE rounding-mode evidence");
#endif
    auto x=floats(argv[4],n);std::vector<float> result;
    if(op=="dot" || op=="q8dot") {
        if(n%32 || std::string(argv[5])!="-")throw std::runtime_error("projection block/C input differs");
        std::vector<float> w;
        if(op=="dot")w=floats(argv[3],n);
        else {
            auto raw=bytes(argv[3],(n/32)*34);w.resize(n);
            for(size_t i=0;i<n;++i) {
                const size_t at=(i/32)*34;const uint16_t bits=uint16_t(raw[at])|(uint16_t(raw[at+1])<<8);
                const unsigned code=raw[at+2+i%32];w[i]=half(bits)*float(code<128?int(code):int(code)-256);
            }
        }
        result={dot32(w,x)};
    } else if(op=="fma") {
        auto a=floats(argv[3],n),c=floats(argv[5],n);result.resize(n);
        for(size_t i=0;i<n;++i)result[i]=::fmaf(a[i],x[i],c[i]);
    } else throw std::runtime_error("only dot/q8dot/fma implemented; intrinsics unqualified");
    for(float value:result)if(!std::isfinite(value))throw std::runtime_error("nonfinite output preserved as failure");
    if(fs::exists(argv[6]))throw std::runtime_error("new output required");
    std::ofstream out(argv[6],std::ios::binary);out.write(reinterpret_cast<const char*>(result.data()),std::streamsize(result.size()*4));out.close();
    if(!out)throw std::runtime_error("output write failed");
    std::cout<<"{\"schema\":1,\"op\":\""<<op<<"\",\"elements\":"<<n<<",\"output_floats\":"<<result.size()
             <<",\"rounding\":\"FE_TONEAREST\",\"flush_to_zero\":false,\"denormals_are_zero\":false,\"fma\":\"host_fmaf\",\"device_intrinsics_qualified\":false,\"model_math_qualified\":false}"<<std::endl;
    return 0;
} catch(const std::exception& e) {std::cerr<<"HC35_HOST_ERROR "<<e.what()<<std::endl;return 2;}
