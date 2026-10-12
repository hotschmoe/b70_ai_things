#include "strata/kernels/original_mtp53_shifted_checkpoint.hpp"
#include <cstdio>
using namespace strata::kernels::original_mtp53;
OriginalCheckpoint53 cp53(int cells){OriginalCheckpoint53 c;c.source_rid=5;c.source_generation=1;c.cells=cells;c.head_ready=cells!=0;if(cells)c.prefix={198};c.publisher_sha="8087dbb39fc73f79ac069b1debe4069a44b73ada900c8bd761569d43a5a90231";c.draft_contract_sha=std::string(64,'a');c.target_contract_sha=std::string(64,'b');c.committed.resize(8915972);c.working.resize(8915972);c.hidden.resize(40960);if(c.head_ready)c.logits.resize(993280);c.payload_sha=checkpoint_digest53(c);return c;}
int main()try{
 ShiftedOriginalCheckpoint53 c;c.target_inputs={248045};c.lookahead=198;c.current=cp53(1);c.pre_last=cp53(0);c.last_target_hidden.resize(40960);c.envelope_sha=shifted_digest53(c);validate_shifted53(c);
 if(c.pre_last.head_ready || !c.pre_last.logits.empty() || c.current.prefix==c.target_inputs)throw std::runtime_error("true empty/shifted contract lost");
 int negatives=0;auto refuse=[&](auto callback){bool yes=false;try{callback();}catch(const std::runtime_error&){yes=true;}if(!yes)throw std::runtime_error("shifted CPU negative accepted");++negatives;};
 auto wrong=c;wrong.current.prefix=wrong.target_inputs;wrong.current.payload_sha=checkpoint_digest53(wrong.current);refuse([&]{validate_shifted53(wrong);});
 wrong=c;wrong.pre_last.head_ready=true;wrong.pre_last.logits.resize(993280);refuse([&]{validate_shifted53(wrong);});
 wrong=c;wrong.lookahead=8678;refuse([&]{validate_shifted53(wrong);});
 wrong=c;wrong.last_target_hidden[2]=0x80;wrong.last_target_hidden[3]=0x7f;refuse([&]{validate_shifted53(wrong);});
 std::printf("MTP53_SHIFTED_CPU_PASS empty_pre_last_real=1 target_next_prefix_distinct=1 controls=%d model_runtime=0\n",negatives);return 0;
}catch(const std::exception& e){std::fprintf(stderr,"MTP53_SHIFTED_CPU_ERROR %s\n",e.what());return 2;}
