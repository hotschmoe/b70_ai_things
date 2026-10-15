// S62 CPU-only provenance audit. Existing ggml quant math/converters, no SYCL.
#include "ggml.h"
#include <algorithm>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <xmmintrin.h>
static void need(bool x,const char* m){if(!x)throw std::runtime_error(m);}
struct Stats {
 uint64_t normal=0,subnormal=0,poszero=0,negzero=0,inf=0,nan=0;
 void add(uint16_t h){unsigned e=(h>>10)&31,m=h&1023;if(e==31){if(m)++nan;else ++inf;}else if(e){++normal;}else if(m){++subnormal;}else if(h&32768){++negzero;}else ++poszero;}
 void out()const{std::cout<<"{\"normal\":"<<normal<<",\"subnormal\":"<<subnormal<<",\"positive_zero\":"<<poszero<<",\"negative_zero\":"<<negzero<<",\"inf\":"<<inf<<",\"nan\":"<<nan<<"}";}
};
struct Fixture {int index,layer,expert,gt,dt,ne,roff;size_t off,bytes,up,down;};
static std::vector<uint8_t> read(const std::string& p){std::ifstream f(p,std::ios::binary|std::ios::ate);need(bool(f),"open input");const auto n=f.tellg();need(n>=0,"input size");std::vector<uint8_t> b(static_cast<size_t>(n));f.seekg(0);f.read(reinterpret_cast<char*>(b.data()),std::streamsize(b.size()));need(size_t(f.gcount())==b.size(),"short input");f.close();need(bool(f),"close input");return b;}
int main(int argc,char**argv){try{
 need(argc==2,"usage audit_operands S42_EXPORTS");need(std::endian::native==std::endian::little,"little endian required");need(std::fegetround()==FE_TONEAREST,"require FE_TONEAREST");unsigned mxcsr=_mm_getcsr();need((mxcsr&0x8040)==0,"require CPU FTZ/DAZ off");
 std::string dir=argv[1];auto blobs=read(dir+"/weights.bin"),input=read(dir+"/actual_gathered.fp16");need(input.size()%2==0,"input half size");Stats xs;for(size_t i=0;i<input.size();i+=2){uint16_t h;std::memcpy(&h,input.data()+i,2);xs.add(h);}need(xs.nan==0&&xs.inf==0,"nonfinite saved input");
 std::ifstream ts(dir+"/fixtures.tsv");need(bool(ts),"open TSV");std::string line;need(bool(std::getline(ts,line)),"TSV header");std::vector<Fixture> fixtures;while(std::getline(ts,line)){Fixture e{};std::istringstream row(line);row>>e.index>>e.layer>>e.expert>>e.gt>>e.dt>>e.ne>>e.roff>>e.off>>e.bytes>>e.up>>e.down;need(bool(row),"parse TSV");fixtures.push_back(e);}need(ts.eof(),"read TSV");ts.clear();ts.close();need(bool(ts),"close TSV");need(fixtures.size()==16,"require sixteen fixtures");
 Stats totals,scales;std::vector<float> dq(2560),back(2560);std::vector<ggml_fp16_t> half(2560);uint64_t fp32_nonfinite=0,widen_mismatch=0;double max_round_error=0;
 std::cout<<"{\"scope\":\"CPU_GGML_RECONSTRUCTED_OPERANDS_ONLY_NOT_GPU_IDENTITY\",\"rounding_FE_TONEAREST\":true,\"MXCSR\":"<<mxcsr<<",\"CPU_FTZ_DAZ\":false,\"saved_input\":";xs.out();std::cout<<",\"fixtures\":[";
 for(size_t i=0;i<fixtures.size();++i){auto&e=fixtures[i];need(e.index==int(i),"fixture order");need(e.off<=blobs.size()&&e.bytes<=blobs.size()-e.off,"blob bounds");const auto*t=ggml_get_type_traits(static_cast<ggml_type>(e.gt));need(t&&t->to_float&&t->blck_size==256,"ggml trait");size_t rb=2560/t->blck_size*t->type_size;need(e.up==640*rb&&e.down==2*e.up&&e.down<=e.bytes,"GU layout");Stats hs,ss;
  for(int parity=0;parity<2;++parity)for(int row=0;row<640;++row){const uint8_t*w=blobs.data()+e.off+parity*e.up+row*rb;
   for(int k=0;k<10;++k){uint16_t h;std::memcpy(&h,w+k*t->type_size,2);ss.add(h);scales.add(h);}
   t->to_float(w,dq.data(),2560);ggml_fp32_to_fp16_row(dq.data(),half.data(),2560);ggml_fp16_to_fp32_row(half.data(),back.data(),2560);
   for(int k=0;k<2560;++k){hs.add(half[k]);totals.add(half[k]);if(!std::isfinite(dq[k]))++fp32_nonfinite;if(!std::isfinite(back[k]))++widen_mismatch;max_round_error=std::max(max_round_error,std::abs(double(dq[k])-double(back[k])));}
  }
  if(i)std::cout<<',';std::cout<<"{\"index\":"<<e.index<<",\"gu_type\":"<<e.gt<<",\"down_type\":"<<e.dt<<",\"raw_GU_scales\":";ss.out();std::cout<<",\"reconstructed_GU_F16\":";hs.out();std::cout<<'}';
 }
 std::cout<<"],\"raw_GU_scales_total\":";scales.out();std::cout<<",\"reconstructed_GU_F16_total\":";totals.out();std::cout<<",\"dequant_FP32_nonfinite\":"<<fp32_nonfinite<<",\"widen_nonfinite\":"<<widen_mismatch<<",\"max_FP32_to_F16_round_abs\":"<<std::setprecision(17)<<max_round_error<<",\"empirical_GPU_operand_byte_comparison\":false,\"selected_FP64_cells_executed\":0}\n";
 need(!fp32_nonfinite&&!widen_mismatch&&!totals.nan&&!totals.inf&&!scales.nan&&!scales.inf,"nonfinite operand");return 0;
}catch(const std::exception&e){std::cerr<<"S62 CPU audit failure: "<<e.what()<<'\n';return 2;}}
