// S62 selected CPU operand rows only; reuse ggml dequant/RN half/widen.
#include "ggml.h"
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <xmmintrin.h>
namespace fs=std::filesystem;
static void need(bool x,const char*m){if(!x)throw std::runtime_error(m);}
static bool normal(uint16_t h){unsigned e=(h>>10)&31;return e!=0&&e!=31;}
static void at(std::ifstream&f,size_t off,void*p,size_t bytes){f.clear();f.seekg(std::streamoff(off));need(bool(f),"seek compact input");f.read(static_cast<char*>(p),std::streamsize(bytes));need(size_t(f.gcount())==bytes,"short compact input");}
static void close(std::ifstream&f){f.clear();f.close();need(bool(f),"close input");}
static void close(std::ofstream&f){f.close();need(bool(f),"close output");}
int main(int argc,char**argv){try{
 need(argc==4,"usage select_cells EXPORTS SELECTED_TSV NEW_EXISTING_DIR");need(std::fegetround()==FE_TONEAREST,"FE_TONEAREST required");unsigned csr=_mm_getcsr();need((csr&0x8040)==0,"CPU FTZ/DAZ must be off");
 fs::path e=argv[1],out=argv[3];need(fs::is_directory(out),"output directory required");
 for(auto n:{"selected-w.fp16","selected-w.f32","selected-x.f32"})need(!fs::exists(out/n),"preserve old output");
 std::ifstream w(e/"weights.bin",std::ios::binary),x(e/"actual_gathered.fp16",std::ios::binary),ts(argv[2]);need(bool(w)&&bool(x)&&bool(ts),"open compact inputs");
 std::ofstream wh(out/"selected-w.fp16",std::ios::binary),wf(out/"selected-w.f32",std::ios::binary),xf(out/"selected-x.f32",std::ios::binary);need(bool(wh)&&bool(wf)&&bool(xf),"open selected outputs");
 std::vector<float> f(2560),fw(2560),fx(2560);std::vector<ggml_fp16_t> hw(2560),hx(2560);std::vector<uint8_t> raw(1200);std::string line;need(bool(std::getline(ts,line)),"selection header");int count=0;
 while(std::getline(ts,line)){int id,fixture,gu,type;size_t flat,xrow,offset,bytes;std::istringstream s(line);s>>id>>fixture>>flat>>xrow>>gu>>type>>offset>>bytes;need(bool(s)&&id==count&&count<64,"parse selected cell");
  const auto*t=ggml_get_type_traits(static_cast<ggml_type>(type));need(t&&t->to_float&&t->blck_size==256&&10*t->type_size==bytes&&bytes<=raw.size(),"selected type geometry");at(w,offset,raw.data(),bytes);
  for(int k=0;k<10;++k){uint16_t h;std::memcpy(&h,raw.data()+k*t->type_size,2);need(normal(h),"selected raw scale must be normal");}
  t->to_float(raw.data(),f.data(),2560);ggml_fp32_to_fp16_row(f.data(),hw.data(),2560);ggml_fp16_to_fp32_row(hw.data(),fw.data(),2560);
  at(x,xrow*2560*2,hx.data(),2560*2);ggml_fp16_to_fp32_row(hx.data(),fx.data(),2560);
  for(int k=0;k<2560;++k)need(normal(hw[k])&&normal(hx[k])&&std::isfinite(f[k])&&std::isfinite(fw[k])&&std::isfinite(fx[k]),"selected CPU operand class");
  wh.write(reinterpret_cast<const char*>(hw.data()),2560*2);wf.write(reinterpret_cast<const char*>(fw.data()),2560*4);xf.write(reinterpret_cast<const char*>(fx.data()),2560*4);need(bool(wh)&&bool(wf)&&bool(xf),"write selected operands");++count;
 }
 need(ts.eof()&&count>=20&&count<=64,"selection size");close(w);close(x);close(ts);close(wh);close(wf);close(xf);
 std::cout<<"{\"status\":\"CPU_NORMAL_SELECTED_ROWS_EXPORTED\",\"cells\":"<<count<<",\"N\":2560,\"FE_TONEAREST\":true,\"MXCSR\":"<<csr<<",\"CPU_FTZ_DAZ\":false,\"raw_scales_normal\":true,\"CPU_operands_normal\":true,\"GPU_operand_identity_measured\":false}\n";return 0;
}catch(const std::exception&e){std::cerr<<"S62 selected row error: "<<e.what()<<'\n';return 2;}}
