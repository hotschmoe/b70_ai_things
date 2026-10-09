// CPU-only dual check against the frozen official GGML scalar dequantizers.
#include "ggml.h"
#include "ggml-quants.h"
#include <cstdint>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>
int main(int argc,char** argv) {
    if(argc!=4)return 2;
    const std::string kind=argv[1];int width=0,count=0;
    if(kind=="Q8_0"){width=34;count=32;}else if(kind=="Q5_1"){width=24;count=32;}
    else if(kind=="Q4_K"){width=144;count=256;}else if(kind=="Q5_K"){width=176;count=256;}
    else if(kind=="IQ4_NL"){width=18;count=32;}else return 3;
    std::ifstream input(argv[2],std::ios::binary);std::vector<uint8_t> raw((std::istreambuf_iterator<char>(input)),{});
    if(!input.is_open() || raw.size()!=size_t(width))return 4;
    ggml_init_params p{1024*1024,nullptr,true};ggml_context* ctx=ggml_init(p);if(!ctx)return 5;
    std::vector<float> values(count);
    if(kind=="Q8_0")dequantize_row_q8_0(reinterpret_cast<const block_q8_0*>(raw.data()),values.data(),count);
    else if(kind=="Q5_1")dequantize_row_q5_1(reinterpret_cast<const block_q5_1*>(raw.data()),values.data(),count);
    else if(kind=="Q4_K")dequantize_row_q4_K(reinterpret_cast<const block_q4_K*>(raw.data()),values.data(),count);
    else if(kind=="Q5_K")dequantize_row_q5_K(reinterpret_cast<const block_q5_K*>(raw.data()),values.data(),count);
    else dequantize_row_iq4_nl(reinterpret_cast<const block_iq4_nl*>(raw.data()),values.data(),count);
    std::ofstream output(argv[3],std::ios::binary);output.write(reinterpret_cast<char*>(values.data()),count*4);
    ggml_free(ctx);return output?0:6;
}
