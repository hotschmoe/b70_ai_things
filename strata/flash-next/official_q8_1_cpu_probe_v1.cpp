// CPU-only official GGML packet variant, not the native raw-sum/division variant.
#include "ggml.h"
#include "ggml-quants.h"
#include <fstream>
#include <cstring>
#include <iterator>
#include <vector>
int main(int argc,char**argv) {
    if(argc!=3)return 2;std::ifstream input(argv[1],std::ios::binary);
    std::vector<char> raw((std::istreambuf_iterator<char>(input)),{});
    if(!input.is_open()||raw.empty()||raw.size()%128)return 3;
    ggml_init_params params{1048576,nullptr,true};auto*ctx=ggml_init(params);if(!ctx)return 4;
    std::vector<float> values(raw.size()/4);std::memcpy(values.data(),raw.data(),raw.size());
    std::vector<block_q8_1> packets(values.size()/32);
    quantize_row_q8_1_ref(values.data(),packets.data(),values.size());
    std::ofstream out(argv[2],std::ios::binary);out.write(reinterpret_cast<char*>(packets.data()),packets.size()*sizeof(block_q8_1));
    ggml_free(ctx);return out?0:5;
}
