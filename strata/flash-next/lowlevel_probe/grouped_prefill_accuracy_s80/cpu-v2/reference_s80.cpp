// S80 CPU-only full GU reference. Existing ggml quant math/RN half; CBLAS FP64.
#include "ggml.h"
#include "mkl_cblas.h"
#include "mkl_service.h"
#include "reference_bound.hpp"
#include <algorithm>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/resource.h>
#include <xmmintrin.h>
namespace fs=std::filesystem;
constexpr int K=2560,G=1280,FF=640,ROWS=1748,EXPERTS=16;
struct Failure:std::runtime_error {using std::runtime_error::runtime_error;};
static void need(bool x,const std::string& m){if(!x)throw Failure(m);}
static void env_check(){need(std::fegetround()==FE_TONEAREST,"FE_TONEAREST required");need((_mm_getcsr()&0x8040)==0,"CPU FTZ/DAZ must be off");}
static void close(std::ifstream& f){f.clear();f.close();need(bool(f),"input close");}
struct Out {
 std::ofstream f;
 explicit Out(const fs::path& p):f(p,std::ios::binary){need(bool(f),"open output "+p.string());}
 template<class T>void write(const std::vector<T>& v){f.write(reinterpret_cast<const char*>(v.data()),std::streamsize(v.size()*sizeof(T)));need(bool(f),"write output");}
 void finish(){f.close();need(bool(f),"close output");}
};
static void at(std::ifstream& f,size_t off,void* p,size_t n){f.clear();f.seekg(std::streamoff(off));need(bool(f),"input seek");f.read(static_cast<char*>(p),std::streamsize(n));need(size_t(f.gcount())==n,"short input");}
struct Stats {
 uint64_t normal=0,subnormal=0,poszero=0,negzero=0,inf=0,nan=0;
 void add(uint16_t h){unsigned e=(h>>10)&31,m=h&1023;if(e==31){if(m)++nan;else ++inf;}else if(e)++normal;else if(m)++subnormal;else if(h&32768)++negzero;else ++poszero;}
 void json(std::ostream& f)const{f<<"{\"normal\":"<<normal<<",\"subnormal\":"<<subnormal<<",\"positive_zero\":"<<poszero<<",\"negative_zero\":"<<negzero<<",\"inf\":"<<inf<<",\"nan\":"<<nan<<'}';}
 bool finite()const{return inf==0&&nan==0;}
 bool gu_model()const{return finite()&&subnormal==0;}
};
struct Fixture{int index,layer,expert,gt,dt,ne,roff;size_t off,bytes,up,down;};
static std::vector<Fixture> fixtures(const fs::path& p){
 std::ifstream f(p);need(bool(f),"open fixtures");std::string line;need(bool(std::getline(f,line)),"fixtures header");
 need(line=="index\tlayer\texpert\tgu_type\tdown_type\tne\trow_offset\tblob_offset\tblob_bytes\tup_offset\tdown_offset","fixtures exact header");
 std::vector<Fixture> v;size_t off=0;int rows=0;
 while(std::getline(f,line)){Fixture e{};std::istringstream s(line);s>>e.index>>e.layer>>e.expert>>e.gt>>e.dt>>e.ne>>e.roff>>e.off>>e.bytes>>e.up>>e.down;need(bool(s),"fixture parse");s>>std::ws;need(s.eof(),"fixture trailing fields");need(e.index==int(v.size())&&e.roff==rows&&e.off==off&&e.ne>0&&e.ne<=641,"fixture order/bounds");need(e.gt==16||e.gt==17||e.gt==18||e.gt==21||e.gt==22,"GU type");need(e.dt==20||e.dt==42,"down type");v.push_back(e);rows+=e.ne;off+=e.bytes;}
 need(f.eof()&&v.size()==EXPERTS&&rows==ROWS&&off==28160000,"fixture totals");close(f);return v;
}
static __int128 integer(const std::string& s){need(!s.empty(),"integer parse");size_t i=s[0]=='-'?1:0;__int128 v=0;need(i<s.size(),"integer digits");for(;i<s.size();++i){need(s[i]>='0'&&s[i]<='9',"integer digit");v=v*10+(s[i]-'0');}return s[0]=='-'?-v:v;}
static int64_t lattice(uint16_t h){unsigned e=(h>>10)&31,m=h&1023;need(e!=31,"finite half lattice");int64_t v=e?int64_t(1024+m)<<(e-1):m;return h&32768?-v:v;}
struct Cell{int id,fixture,token,col;size_t flat;__int128 dot,abs;double exact;};
static std::vector<Cell> cells(const fs::path& p){std::ifstream f(p);need(bool(f),"selected36 open");std::string line;need(bool(std::getline(f,line)),"selected header");std::vector<Cell> v;while(std::getline(f,line)){Cell c{};std::string d,a,h;std::istringstream s(line);s>>c.id>>c.fixture>>c.token>>c.col>>c.flat>>d>>a>>h;need(bool(s)&&c.id==int(v.size()),"selected parse");s>>std::ws;need(s.eof(),"selected trailing");c.dot=integer(d);c.abs=integer(a);c.exact=std::stod(h);need(std::isfinite(c.exact)&&c.fixture>=0&&c.fixture<16&&c.token>=0&&c.col>=0&&c.col<G,"selected bounds");v.push_back(c);}need(f.eof()&&v.size()==36,"selected36 count");close(f);return v;}
static void dgemm(const std::vector<double>& x,const std::vector<double>& w,std::vector<double>& y,int ne){env_check();cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,ne,G,K,1.0,x.data(),K,w.data(),K,0.0,y.data(),G);env_check();}
static void row_half(const uint8_t* raw,const ggml_type_traits* t,int n,ggml_fp16_t* h,float* dq,float* widened){t->to_float(raw,dq,n);for(int k=0;k<n;++k)need(std::isfinite(dq[k]),"nonfinite CPU dequant");ggml_fp32_to_fp16_row(dq,h,n);ggml_fp16_to_fp32_row(h,widened,n);}
int main(int argc,char**argv){try{
 need(argc==4,"usage reference_s80 COMPACT_INPUT_DIR SELECTED36_TSV NEW_OUTPUT_DIR");need(std::endian::native==std::endian::little,"little-endian required");static_assert(sizeof(double)==8&&sizeof(float)==4&&sizeof(ggml_fp16_t)==2&&sizeof(MKL_INT)==4);
 env_check();need(!std::getenv("MKL_CBWR"),"MKL_CBWR must remain absent");mkl_set_dynamic(0);mkl_set_num_threads(4);need(mkl_get_max_threads()==4,"MKL threads4");
 const fs::path input=argv[1],out=argv[3];need(fs::create_directory(out),"fresh reference output required");
 const auto fixture=fixtures(input/"fixtures.tsv");const auto selected=cells(argv[2]);
 need(fs::file_size(input/"weights.bin")==28160000&&fs::file_size(input/"actual_gathered.fp16")==size_t(ROWS)*K*2,"compact sizes");
 std::ifstream raw(input/"weights.bin",std::ios::binary),xf(input/"actual_gathered.fp16",std::ios::binary);need(bool(raw)&&bool(xf),"compact opens");
 std::vector<ggml_fp16_t> input_half(size_t(ROWS)*K);at(xf,0,input_half.data(),input_half.size()*2);close(xf);
 Stats xs,gs,ds,scale;for(auto h:input_half)xs.add(h);need(xs.gu_model(),"input nonfinite/subnormal STOP");
 Out gu(out/"gu-planes.fp16"),down(out/"down-planes.fp16"),Tfile(out/"gu-T64.f64"),Afile(out/"gu-Aupper.f64"),Bfile(out/"gu-B32.f64"),Efile(out/"gu-E64.f64"),boundfile(out/"gu-bound.f64"),changed(out/"actual-gathered-changed.fp16");
 {auto signflip=input_half;for(auto& h:signflip)h^=uint16_t(0x8000);changed.write(signflip);changed.finish();}
 const auto g32=s80ref::gamma(K,-24),g64=s80ref::gamma(K,-53);uint64_t values=0,zeros=0,sanity=0;double max_sanity_error=0;bool negative_checked=false,zero_dgemm=false;size_t negative_flat=0;double negative_y=0,negative_ref=0,negative_bound=0,negative_endpoint=0;
 std::vector<float> dq(K),wide(K);std::vector<ggml_fp16_t> hrow(K);
 for(const auto& e:fixture){
  env_check();std::vector<uint8_t> blob(e.bytes);at(raw,e.off,blob.data(),blob.size());
  const auto* gt=ggml_get_type_traits(static_cast<ggml_type>(e.gt));const auto* dt=ggml_get_type_traits(static_cast<ggml_type>(e.dt));
  need(gt&&gt->to_float&&gt->blck_size==256,"CPU GU quant trait geometry");
  need(dt&&dt->to_float&&dt->blck_size==(e.dt==20?32:64)&&FF%dt->blck_size==0,"CPU down quant trait geometry");
  const size_t rb=size_t(K/256)*gt->type_size,drb=size_t(FF/dt->blck_size)*dt->type_size;need(e.up==640*rb&&e.down==2*e.up&&e.down+2560*drb==e.bytes,"quant layout");
  std::vector<ggml_fp16_t> wh(size_t(G)*K);std::vector<double> w(wh.size()),x(size_t(e.ne)*K),T(size_t(e.ne)*G),A(T.size()),B(T.size()),E(T.size()),bound(T.size());
  for(int col=0;col<G;++col){const uint8_t* qrow=blob.data()+(col&1)*e.up+(col/2)*rb;
   for(int z=0;z<K/256;++z){uint16_t h;std::memcpy(&h,qrow+z*gt->type_size,2);scale.add(h);}
   row_half(qrow,gt,K,wh.data()+size_t(col)*K,dq.data(),wide.data());
   for(int k=0;k<K;++k){const auto h=wh[size_t(col)*K+k];gs.add(h);need(((h>>10)&31)!=31&&(((h>>10)&31)!=0||(h&1023)==0),"CPU GU half nonfinite/subnormal STOP");w[size_t(col)*K+k]=double(wide[k]);}
  }
  gu.write(wh);
  for(int t=0;t<e.ne;++t){ggml_fp16_to_fp32_row(input_half.data()+size_t(e.roff+t)*K,wide.data(),K);for(int k=0;k<K;++k)x[size_t(t)*K+k]=double(wide[k]);}
  dgemm(x,w,T,e.ne);
  // Independent exact half-lattice conformance, not a fitted numerical tolerance.
  for(const auto& c:selected)if(c.fixture==e.index){need(c.token<e.ne&&c.flat==size_t(e.roff+c.token)*G+c.col,"selected fixture mapping");__int128 num=0,absnum=0;
   for(int k=0;k<K;++k){const __int128 v=__int128(lattice(wh[size_t(c.col)*K+k]))*lattice(input_half[size_t(e.roff+c.token)*K+k]);num+=v;absnum+=v<0?-v:v;}
   need(num==c.dot&&absnum==c.abs,"S62 exact numerator conformance");const double exact=std::ldexp(double(num),-48);need(exact==c.exact&&__int128(std::ldexp(exact,48))==num,"S62 exact double conformance");
  }
  for(auto& v:w)v=std::abs(v);for(auto& v:x)v=std::abs(v);dgemm(x,w,A,e.ne);
  if(!zero_dgemm){std::vector<double> zx(K,0.0),zy(G,std::numeric_limits<double>::quiet_NaN());dgemm(zx,w,zy,1);for(double z:zy)need(z==0.0,"mathematical zero DGEMM");auto z=s80ref::bounds(0,g32,g64);need(z.Aupper==0&&z.total==0&&s80ref::accepts(0,0,0)&&s80ref::accepts(-0.0,0,0)&&!s80ref::accepts(double(std::numeric_limits<float>::denorm_min()),0,0),"zero negative-control gate");zero_dgemm=true;}
  for(size_t i=0;i<T.size();++i){need(std::isfinite(T[i]),"reference finite");const auto v=s80ref::bounds(A[i],g32,g64);if(A[i]==0){need(T[i]==0,"zero A requires zero T");++zeros;}A[i]=v.Aupper;B[i]=v.B32;E[i]=v.E64;bound[i]=v.total;
   if(!negative_checked&&v.total>0){need(s80ref::accepts(double(float(T[i])),T[i],v.total),"rounded-reference positive control");const double endpoint=s80ref::up(T[i]+v.total);float bad=float(endpoint);if(double(bad)<=endpoint)bad=std::nextafter(bad,std::numeric_limits<float>::infinity());need(std::isfinite(bad)&&double(bad)>endpoint&&!s80ref::accepts(double(bad),T[i],v.total),"outside-bound finite negative control");negative_flat=size_t(e.roff)*G+i;negative_y=double(bad);negative_ref=T[i];negative_bound=v.total;negative_endpoint=endpoint;negative_checked=true;}
  }
  for(const auto& c:selected)if(c.fixture==e.index){const size_t i=size_t(c.token)*G+c.col;const double error=s80ref::residual_upper(T[i],c.exact);need(error<=E[i],"DGEMM selected reference uncertainty");const double exactA=std::ldexp(double(c.abs),-48);need(c.abs==0?A[i]==0:A[i]>=s80ref::up(exactA),"selected Aupper encloses exact absolute sum");max_sanity_error=std::max(max_sanity_error,error);++sanity;}
  Tfile.write(T);Afile.write(A);Bfile.write(B);Efile.write(E);boundfile.write(bound);values+=T.size();
  // Optional CPU down plane identity only: no CPU H or down target.
  std::vector<ggml_fp16_t> dh(size_t(2560)*FF);for(int row=0;row<2560;++row){row_half(blob.data()+e.down+size_t(row)*drb,dt,FF,dh.data()+size_t(row)*FF,dq.data(),wide.data());for(int k=0;k<FF;++k){auto h=dh[size_t(row)*FF+k];ds.add(h);need(((h>>10)&31)!=31,"down CPU half finite");}}down.write(dh);
  env_check();std::cout<<"expert "<<e.index<<" rows "<<e.ne<<" complete\n";
 }
 close(raw);for(Out* f:{&gu,&down,&Tfile,&Afile,&Bfile,&Efile,&boundfile})f->finish();
 need(values==2237440&&sanity==36&&negative_checked&&zero_dgemm&&gs.gu_model()&&ds.finite()&&scale.finite(),"full reference closure");env_check();rusage usage{};need(getrusage(RUSAGE_SELF,&usage)==0,"getrusage");need(usage.ru_maxrss<1024L*1024L,"peak RSS must be under1GiB");
 char version[256]{};mkl_get_version_string(version,sizeof(version));std::ofstream vf(out/"mkl-version.txt");need(bool(vf),"version open");vf<<version<<'\n';need(bool(vf),"version write");vf.close();need(bool(vf),"version close");std::ofstream core(out/"reference-core.json");need(bool(core),"core open");core<<std::setprecision(17)<<"{\"status\":\"PASS_CPU_REFERENCE_ONLY_GPU_OPERANDS_UNMEASURED\",\"experts\":16,\"rows\":1748,\"GU_columns\":1280,\"K\":2560,\"values\":"<<values<<",\"threads\":4,\"max_rss_kib\":"<<usage.ru_maxrss<<",\"FE_TONEAREST\":true,\"MXCSR\":"<<_mm_getcsr()<<",\"FTZ_DAZ\":false,\"gamma32\":["<<g32.lo<<','<<g32.hi<<"],\"gamma64\":["<<g64.lo<<','<<g64.hi<<"],\"exact_zero_A_values\":"<<zeros<<",\"selected36_exact_numerators\":true,\"selected36_E64_conformance\":true,\"max_selected_reference_residual_upper\":"<<max_sanity_error<<",\"zero_DGEMM_and_zero_negative_control\":true,\"zero_DGEMM_cells\":1280,\"outside_bound_float_negative_control\":true,\"negative_control_flat_index\":"<<negative_flat<<",\"negative_control_y\":"<<negative_y<<",\"negative_control_reference\":"<<negative_ref<<",\"negative_control_bound\":"<<negative_bound<<",\"negative_control_endpoint_up\":"<<negative_endpoint<<",\"GPU_plane_identity_measured\":false,\"ideal_H_down_reference\":false,\"input_half_classes\":";xs.json(core);core<<",\"GU_half_classes\":";gs.json(core);core<<",\"down_half_classes\":";ds.json(core);core<<",\"raw_GU_scale_classes\":";scale.json(core);core<<"}\n";need(bool(core),"core write");core.close();need(bool(core),"core close");
 std::cout<<"PASS_CPU_REFERENCE_ONLY all2237440 GU cells; GPU operand identity UNMEASURED\n";return 0;
}catch(const std::exception& e){std::cerr<<"S80 CPU reference failure: "<<e.what()<<'\n';return 2;}}
