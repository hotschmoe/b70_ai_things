// Payload for draft0019: bounded host-only observations, never cache policy.
#pragma once
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>
#include <sys/stat.h>
#include <unistd.h>
namespace strata::core::prefix_lifecycle {
class Sha256 {
    uint32_t h_[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    uint8_t block_[64]{};size_t used_=0;uint64_t bytes_=0;
    static uint32_t rotr(uint32_t x,unsigned n){return (x>>n)|(x<<(32-n));}
    void compress(){
        static constexpr uint32_t k[64]={0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
            0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
            0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
            0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
            0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
            0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
            0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
            0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
        uint32_t w[64];for(unsigned i=0;i<16;++i)w[i]=(uint32_t(block_[4*i])<<24)|(uint32_t(block_[4*i+1])<<16)|(uint32_t(block_[4*i+2])<<8)|block_[4*i+3];
        for(unsigned i=16;i<64;++i){uint32_t x=w[i-15],y=w[i-2];w[i]=w[i-16]+(rotr(x,7)^rotr(x,18)^(x>>3))+w[i-7]+(rotr(y,17)^rotr(y,19)^(y>>10));}
        uint32_t a=h_[0],b=h_[1],c=h_[2],d=h_[3],e=h_[4],f=h_[5],g=h_[6],h=h_[7];
        for(unsigned i=0;i<64;++i){uint32_t t1=h+(rotr(e,6)^rotr(e,11)^rotr(e,25))+((e&f)^(~e&g))+k[i]+w[i];uint32_t t2=(rotr(a,2)^rotr(a,13)^rotr(a,22))+((a&b)^(a&c)^(b&c));h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;}
        const uint32_t v[8]={a,b,c,d,e,f,g,h};for(unsigned i=0;i<8;++i)h_[i]+=v[i];
    }
public:
    void update(const uint8_t* p,size_t n){bytes_+=n;for(size_t i=0;i<n;++i){block_[used_++]=p[i];if(used_==64){compress();used_=0;}}}
    void finish(char out[65]){uint64_t bits=bytes_*8;block_[used_++]=0x80;if(used_>56){while(used_<64)block_[used_++]=0;compress();used_=0;}while(used_<56)block_[used_++]=0;for(int i=7;i>=0;--i)block_[used_++]=uint8_t(bits>>(i*8));compress();for(unsigned i=0;i<8;++i)std::snprintf(out+8*i,9,"%08x",h_[i]);out[64]=0;}
};
template<class Token>inline void token_digest(const std::vector<Token>& ids,char out[65]){Sha256 s;for(auto id:ids){uint32_t t=uint32_t(id);uint8_t b[4]={uint8_t(t),uint8_t(t>>8),uint8_t(t>>16),uint8_t(t>>24)};s.update(b,4);}s.finish(out);}
struct Context {bool active=false;uint64_t request=0,sequence=0;size_t events=0;char input_sha[65]{};};
inline Context& context(){static Context c;return c;}
inline std::mutex& mutex(){static std::mutex m;return m;}
inline bool enabled(){static const bool on=[] {const char* p=std::getenv("STRATA_PREFIX_LIFECYCLE_DIAG");return p&&!std::strcmp(p,"1");}();return on;}
inline bool active(){return enabled()&&context().active;}
inline bool acquire(){auto& c=context();if(!c.active)return false;if(c.events++<4096)return true;if(c.events==4097)std::fprintf(stderr,"PCL {\"event\":\"skip\",\"pid\":%ld,\"request\":%llu,\"reason\":\"event_quota\"}\n",long(getpid()),(unsigned long long)c.request);return false;}
inline void begin(const std::vector<int64_t>& ids,bool supported){if(!enabled())return;std::lock_guard<std::mutex> lock(mutex());auto& c=context();c.active=false;c.events=0;const char* arm=std::getenv("STRATA_PREFIX_DIAG_ARM");struct stat st;if(!arm||stat(arm,&st)||!S_ISREG(st.st_mode))return;if(!supported||ids.empty()||ids.size()>8192||++c.sequence>64)return;c.active=true;c.request=c.sequence;token_digest(ids,c.input_sha);std::fprintf(stderr,"PCL {\"event\":\"begin\",\"pid\":%ld,\"request\":%llu,\"input_sha256_le32\":\"%s\",\"tokens\":%zu,\"scope\":\"serial_text\"}\n",long(getpid()),(unsigned long long)c.request,c.input_sha,ids.size());}
struct Registry {std::array<const void*,256> pointers{};std::array<uint64_t,256> ids{};uint64_t next=0;};
inline Registry& registry(){static Registry r;return r;}
inline uint64_t instance(const void* p){if(!active())return 0;auto& r=registry();for(size_t i=0;i<r.pointers.size();++i)if(r.pointers[i]==p)return r.ids[i];return 0;}
inline uint64_t admit_instance(const void* p){if(!active())return 0;auto& r=registry();for(size_t i=0;i<r.pointers.size();++i)if(r.pointers[i]==p||!r.pointers[i]){r.pointers[i]=p;return r.ids[i]=++r.next;}return 0;}
inline void forget_instance(const void* p){if(!active())return;auto& r=registry();for(size_t i=0;i<r.pointers.size();++i)if(r.pointers[i]==p){r.pointers[i]=nullptr;r.ids[i]=0;return;}}
struct RegistryCapture {std::array<const void*,256> pointers{};std::array<uint64_t,256> ids{};size_t count=0;};
template<class Entries>inline RegistryCapture capture_instances(const Entries& entries){RegistryCapture out;if(!active())return out;out.count=entries.size()<256?entries.size():256;for(size_t i=0;i<out.count;++i){out.pointers[i]=&entries[i];out.ids[i]=instance(out.pointers[i]);}return out;}
inline void restore_instance(const void* p,uint64_t id){if(!active()||!id)return;auto& r=registry();for(size_t i=0;i<r.pointers.size();++i)if(r.pointers[i]==p||!r.pointers[i]){r.pointers[i]=p;r.ids[i]=id;return;}}
template<class Entries>inline void rebind_after_erase(const Entries& entries,size_t removed,const RegistryCapture& before){if(!active())return;for(size_t i=0;i<before.count;++i)forget_instance(before.pointers[i]);for(size_t i=0;i<entries.size()&&i<255;++i){size_t old=i<removed?i:i+1;if(old<before.count)restore_instance(&entries[i],before.ids[old]);}}
inline void number(Sha256& s,uint64_t n){uint8_t b[8];for(unsigned i=0;i<8;++i)b[i]=uint8_t(n>>(8*i));s.update(b,8);}
template<class Token>inline void digest_tokens(Sha256& s,const std::vector<Token>& ids){number(s,ids.size());for(auto id:ids)number(s,uint32_t(id));}
template<class Image>inline void digest_image(Sha256& s,const Image& image){for(auto value:image.geometry)number(s,uint64_t(value));number(s,image.layer_lo);number(s,image.layer_hi);number(s,image.cvec);digest_tokens(s,image.live.ids);number(s,image.live.imgs.size());for(const auto& im:image.live.imgs){number(s,im.start);number(s,im.hash);}number(s,image.checkpoints.size());for(const auto& cp:image.checkpoints){digest_tokens(s,cp.ids);number(s,cp.pinned);number(s,cp.imgs.size());for(const auto& im:cp.imgs){number(s,im.start);number(s,im.hash);}}number(s,image.stage_images.size());for(const auto& stage:image.stage_images)digest_image(s,stage);}
struct Snapshot {char key[65]{},live_key[65]{};uint64_t instance=0;size_t tokens=0,bytes=0,stages=0,checkpoints=0;bool pinned=false;std::array<int64_t,16> bounds{};};
template<class Image>inline Snapshot describe(const Image& image){Snapshot d;if(!active())return d;Sha256 hash;digest_image(hash,image);hash.finish(d.key);d.instance=instance(&image);const auto* ids=&image.live.ids;if(ids->empty())for(const auto& cp:image.checkpoints)if(cp.ids.size()>ids->size())ids=&cp.ids;token_digest(*ids,d.live_key);d.tokens=ids->size();d.bytes=image.bytes();d.pinned=image.pinned();d.checkpoints=image.checkpoints.size();d.stages=image.stage_images.size()+1;d.bounds[0]=image.layer_lo;d.bounds[1]=image.layer_hi;for(size_t i=0;i<image.stage_images.size()&&i<7;++i){d.bounds[2*i+2]=image.stage_images[i].layer_lo;d.bounds[2*i+3]=image.stage_images[i].layer_hi;}return d;}
inline void cache(const char* action,const Snapshot& d,size_t bytes,size_t entries,size_t budget,size_t slots,size_t held=0){
    if(!active())return;std::lock_guard<std::mutex> lock(mutex());if(!acquire())return;const auto& c=context();
    std::fprintf(stderr,"PCL {\"event\":\"cache\",\"pid\":%ld,\"request\":%llu,\"action\":\"%s\",\"snapshot_metadata_sha256\":\"%s\",\"snapshot_instance\":%llu,\"identity_observed\":%s,\"source_tokens_sha256_le32\":\"%s\",\"snapshot_tokens\":%zu,\"snapshot_bytes\":%zu,\"pinned\":%s,\"checkpoints\":%zu,\"retained_bytes\":%zu,\"entries\":%zu,\"budget\":%zu,\"slots\":%zu,\"held\":%zu,\"stage_ranges\":[",long(getpid()),(unsigned long long)c.request,action,d.key,(unsigned long long)d.instance,d.instance?"true":"false",d.live_key,d.tokens,d.bytes,d.pinned?"true":"false",d.checkpoints,bytes,entries,budget,slots,held);
    for(size_t i=0;i<d.stages&&i<8;++i)std::fprintf(stderr,"%s[%lld,%lld]",i?",":"",(long long)d.bounds[2*i],(long long)d.bounds[2*i+1]);
    std::fprintf(stderr,"],\"stage_descriptor_truncated\":%s}\n",d.stages>8?"true":"false");
}
template<class Token>inline void state(const char* event,const std::vector<Token>& ids,bool updated,bool reusable,const char* phase,const char* finish){
    if(!active())return;std::lock_guard<std::mutex> lock(mutex());if(!acquire())return;char key[65];token_digest(ids,key);const auto& c=context();
    std::fprintf(stderr,"PCL {\"event\":\"%s\",\"pid\":%ld,\"request\":%llu,\"phase\":\"%s\",\"finish\":\"%s\",\"published\":%s,\"chain_updated\":%s,\"live_reusable\":%s,\"tokens\":%zu,\"sha256_le32\":\"%s\",\"ids\":[",event,long(getpid()),(unsigned long long)c.request,phase,finish,(updated&&reusable)?"true":"false",updated?"true":"false",reusable?"true":"false",ids.size(),key);
    if(ids.size()<=8192)for(size_t i=0;i<ids.size();++i)std::fprintf(stderr,"%s%lld",i?",":"",(long long)ids[i]);
    std::fprintf(stderr,"],\"ids_truncated\":%s}\n",ids.size()>8192?"true":"false");
}
inline void stage(const char* phase,int device,int64_t lb,int64_t le,int64_t lo,int64_t hi,bool complete){if(!active())return;std::lock_guard<std::mutex> lock(mutex());if(!acquire())return;const auto& c=context();std::fprintf(stderr,"PCL {\"event\":\"stage_span\",\"pid\":%ld,\"request\":%llu,\"phase\":\"%s\",\"device\":%d,\"lb\":%lld,\"le\":%lld,\"lo\":%lld,\"hi\":%lld,\"complete\":%s,\"commit_proven\":false}\n",long(getpid()),(unsigned long long)c.request,phase,device,(long long)lb,(long long)le,(long long)lo,(long long)hi,complete?"true":"false");}
}
