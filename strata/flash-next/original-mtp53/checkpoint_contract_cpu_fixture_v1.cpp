#include "strata/kernels/original_mtp53_checkpoint_contract.hpp"
#include <cstdio>
using namespace strata::kernels::original_mtp53;
void require53(bool yes){if(!yes)throw std::runtime_error("CPU checkpoint contract control failed");}
int main()try{
 OriginalCheckpoint53 c;c.source_rid=7;c.source_generation=3;c.cells=2;c.prefix={248045,198};c.publisher_sha="8087dbb39fc73f79ac069b1debe4069a44b73ada900c8bd761569d43a5a90231";c.draft_contract_sha=std::string(64,'a');c.target_contract_sha=std::string(64,'b');
 c.committed.resize(8915972);c.working.resize(8915972);c.hidden.resize(40960);c.logits.resize(993280);c.committed[8915960]=0x80;c.working[173]=0xff;c.hidden[3]=0x80;
 c.payload_sha=checkpoint_digest53(c);auto bytes=serialize_checkpoint53(c);auto restored=deserialize_checkpoint53(bytes,c.payload_sha);
 require53(restored.committed==c.committed && restored.working==c.working && restored.hidden==c.hidden && restored.logits==c.logits && restored.prefix==c.prefix && restored.cells==c.cells);
 int negatives=0;
 auto refuse=[&](auto operation){bool yes=false;try{operation();}catch(const std::runtime_error&){yes=true;}require53(yes);++negatives;};
 auto corrupt=bytes;corrupt[corrupt.size()-65]^=1;refuse([&]{deserialize_checkpoint53(corrupt,c.payload_sha);});
 auto truncated=bytes;truncated.pop_back();refuse([&]{deserialize_checkpoint53(truncated,c.payload_sha);});
 auto extra=bytes;extra.push_back(0);refuse([&]{deserialize_checkpoint53(extra,c.payload_sha);});
 auto forged=c;forged.committed[0]^=1;forged.payload_sha=checkpoint_digest53(forged);auto forgedbytes=serialize_checkpoint53(forged);refuse([&]{deserialize_checkpoint53(forgedbytes,c.payload_sha);});
 auto bad=c;bad.prefix.pop_back();refuse([&]{checkpoint_digest53(bad);});
 bad=c;bad.cells=4097;refuse([&]{checkpoint_digest53(bad);});
 bad=c;bad.logits[0]=0;bad.logits[1]=0;bad.logits[2]=0x80;bad.logits[3]=0x7f;refuse([&]{checkpoint_digest53(bad);});
 std::printf("MTP53_CHECKPOINT_CPU_PASS controls=%d exact_LE_full_banks_hidden_logits_prefix=1 independently_pinned_hash=1 native_runtime=0\n",negatives);return 0;
}catch(const std::exception& e){std::fprintf(stderr,"MTP53_CHECKPOINT_CPU_ERROR %s\n",e.what());return 2;}
