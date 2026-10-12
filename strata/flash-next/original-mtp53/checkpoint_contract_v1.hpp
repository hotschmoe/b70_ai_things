#pragma once
// Bounded complete original recurrence serialization. No SYCL/device operations.
#include "strata/core/prefix_lifecycle.hpp"
#include <bit>
#include <algorithm>
#include <cmath>
#include <string>
#include <stdexcept>
namespace strata::kernels::original_mtp53 {
struct OriginalCheckpoint53 {
 uint64_t source_rid=0,source_generation=0,cells=0;
 std::string publisher_sha,draft_contract_sha,target_contract_sha,payload_sha;
 std::vector<int32_t> prefix;
 std::vector<uint8_t> committed,working,hidden,logits;
};
inline bool checkpoint_sha53(const std::string& s){return s.size()==64 && std::all_of(s.begin(),s.end(),[](char c){return(c>='0'&&c<='9')||(c>='a'&&c<='f');});}
inline void validate_checkpoint53(const OriginalCheckpoint53& c){
 static_assert(std::endian::native==std::endian::little && sizeof(float)==4,"original53 checkpoint requires original LE F32 storage");
 if(!c.source_rid || !c.source_generation || c.cells<1 || c.cells>4096 || c.prefix.size()!=c.cells || c.publisher_sha!="8087dbb39fc73f79ac069b1debe4069a44b73ada900c8bd761569d43a5a90231" || !checkpoint_sha53(c.draft_contract_sha) || !checkpoint_sha53(c.target_contract_sha) || c.committed.size()!=8915972 || c.working.size()!=8915972 || c.hidden.size()!=40960 || c.logits.size()!=993280)throw std::runtime_error("original checkpoint complete schema/identity differs");
 for(int32_t t:c.prefix)if(t<0 || t>=248320)throw std::runtime_error("original checkpoint accepted prefix invalid");
 for(size_t at=0;at<c.logits.size();at+=4){float f;std::memcpy(&f,c.logits.data()+at,4);if(!std::isfinite(f))throw std::runtime_error("original checkpoint current head nonfinite");}
}
inline std::vector<uint8_t> checkpoint_body53(const OriginalCheckpoint53& c){
 validate_checkpoint53(c);std::vector<uint8_t> out;out.reserve(20<<20);
 auto num=[&](uint64_t n){for(int i=0;i<8;++i)out.push_back(uint8_t(n>>(8*i)));};
 auto bytes=[&](const uint8_t* p,size_t n){num(n);out.insert(out.end(),p,p+n);};
 const char* magic="ORIGINAL34-CP1-LE4096";bytes(reinterpret_cast<const uint8_t*>(magic),std::strlen(magic));
 num(c.source_rid);num(c.source_generation);num(c.cells);
 for(const auto* s:{&c.publisher_sha,&c.draft_contract_sha,&c.target_contract_sha})bytes(reinterpret_cast<const uint8_t*>(s->data()),s->size());
 num(c.prefix.size());for(int32_t t:c.prefix){uint32_t v=uint32_t(t);for(int i=0;i<4;++i)out.push_back(uint8_t(v>>(8*i)));}
 for(const auto* b:{&c.committed,&c.working,&c.hidden,&c.logits})bytes(b->data(),b->size());return out;
}
inline std::string checkpoint_digest53(const OriginalCheckpoint53& c){auto bytes=checkpoint_body53(c);strata::core::prefix_lifecycle::Sha256 hash;hash.update(bytes.data(),bytes.size());char digest[65];hash.finish(digest);return digest;}
inline std::vector<uint8_t> serialize_checkpoint53(const OriginalCheckpoint53& c){auto out=checkpoint_body53(c);if(c.payload_sha!=checkpoint_digest53(c))throw std::runtime_error("original checkpoint payload hash differs");out.insert(out.end(),c.payload_sha.begin(),c.payload_sha.end());return out;}
inline OriginalCheckpoint53 deserialize_checkpoint53(const std::vector<uint8_t>& input,const std::string& expected_payload_sha){
 if(!checkpoint_sha53(expected_payload_sha))throw std::runtime_error("original checkpoint independently pinned digest required");
 if(input.size()>20u<<20)throw std::runtime_error("original checkpoint serialized bound");size_t at=0;
 auto num=[&](){if(at>input.size() || input.size()-at<8)throw std::runtime_error("original checkpoint truncated number");uint64_t v=0;for(int i=0;i<8;++i)v|=uint64_t(input[at++])<<(8*i);return v;};
 auto bytes=[&](size_t cap){uint64_t n=num();if(n>cap || at>input.size() || n>input.size()-at)throw std::runtime_error("original checkpoint truncated/bounded bytes");std::vector<uint8_t> b(input.begin()+at,input.begin()+at+size_t(n));at+=size_t(n);return b;};
 auto magic=bytes(32);if(std::string(magic.begin(),magic.end())!="ORIGINAL34-CP1-LE4096")throw std::runtime_error("original checkpoint schema magic differs");
 OriginalCheckpoint53 c;c.source_rid=num();c.source_generation=num();c.cells=num();
 for(auto* s:{&c.publisher_sha,&c.draft_contract_sha,&c.target_contract_sha}){auto b=bytes(64);*s=std::string(b.begin(),b.end());}
 uint64_t n=num();if(n>4096 || at>input.size() || n*4>input.size()-at)throw std::runtime_error("original checkpoint prefix bound");for(uint64_t i=0;i<n;++i){uint32_t t=0;for(int j=0;j<4;++j)t|=uint32_t(input[at++])<<(8*j);c.prefix.push_back(int32_t(t));}
 c.committed=bytes(8915972);c.working=bytes(8915972);c.hidden=bytes(40960);c.logits=bytes(993280);
 if(input.size()-at!=64)throw std::runtime_error("original checkpoint trailing bytes/hash missing");c.payload_sha=std::string(input.begin()+at,input.end());validate_checkpoint53(c);if(c.payload_sha!=expected_payload_sha || !checkpoint_sha53(c.payload_sha) || c.payload_sha!=checkpoint_digest53(c))throw std::runtime_error("original checkpoint hash mismatch");return c;
}
}
