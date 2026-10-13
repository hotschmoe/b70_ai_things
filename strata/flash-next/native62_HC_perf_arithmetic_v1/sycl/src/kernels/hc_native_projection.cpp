// Qualification-first native hyper-connection projections. Not a fused HC implementation.
#include "strata/kernels/hc_native_projection.hpp"
#include "strata/sycl_queue.hpp"
#include <sycl/sycl.hpp>
#include <cstdlib>
#include <cstring>

namespace strata::kernels {
namespace {
// Fixed subgroup reduction tree: reproducible within this kernel, not bitwise CPU-serial summation.
float sum32(sycl::sub_group group, float value) {
    for (unsigned mask = 16; mask; mask >>= 1) {
        value += sycl::permute_group_by_xor(group, value, mask);
    }
    return value;
}

template<bool Q8> class HcProjectionKernel;
template<bool Q8>
void launch(sycl::queue & queue, const void * weights, const float * input, float * output,
            int k, int m, int tokens) {
    // One 32-lane subgroup per output row; no shared memory, atomics, or global scratch.
    const size_t rows = size_t(m)*tokens;
    queue.parallel_for<HcProjectionKernel<Q8>>(
        sycl::nd_range<1>(sycl::range<1>(rows*32), sycl::range<1>(32)),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
            const size_t row_token = item.get_group_linear_id();
            const int row = int(row_token % size_t(m));
            const int token = int(row_token / size_t(m));
            const unsigned lane = item.get_local_linear_id();
            const float * x = input + size_t(token)*k;
            float acc = 0.0f;
            for (int column = int(lane); column < k; column += 32) {
                float weight;
                if constexpr (Q8) {
                    const auto * blocks = static_cast<const uint8_t *>(weights);
                    const size_t offset = (size_t(row)*(k/32) + column/32)*34;
                    const uint16_t bits = uint16_t(blocks[offset]) | (uint16_t(blocks[offset+1]) << 8);
                    const float scale = float(sycl::bit_cast<sycl::half>(bits));
                    // Decode signed codes without relying on implementation-defined unsigned->int8 conversion.
                    const unsigned raw = blocks[offset+2+column%32];
                    const int code = raw < 128 ? int(raw) : int(raw)-256;
                    weight = scale*float(code);
                } else {
                    weight = static_cast<const float *>(weights)[size_t(row)*k+column];
                }
                acc = sycl::fma(weight, x[column], acc);
            }
            const float result = sum32(item.get_sub_group(), acc);
            if (lane == 0) { output[row_token] = result; }
        });
}

// Token reuse only: each output keeps its original lane FMA order and sum32.
// No shared/local memory, atomics, barriers, host callback, quantization, or wait.
template<int Tile> class HcQ8TokenReuseKernel;
template<int Tile>
void launch_q8_token_reuse(sycl::queue& queue,const uint8_t* weights,
        const float* input,float* output,int k,int m,int tokens) {
    const size_t blocks=(size_t(tokens)+Tile-1)/Tile;
    queue.parallel_for<HcQ8TokenReuseKernel<Tile>>(
        sycl::nd_range<1>(sycl::range<1>(size_t(m)*blocks*32),sycl::range<1>(32)),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
            const size_t group=item.get_group_linear_id();
            const int row=int(group%size_t(m)),first=int(group/size_t(m))*Tile;
            const unsigned lane=item.get_local_linear_id();float acc[Tile]={};
            for(int column=int(lane);column<k;column+=32) {
                const size_t offset=(size_t(row)*(k/32)+column/32)*34;
                const uint16_t bits=uint16_t(weights[offset])|(uint16_t(weights[offset+1])<<8);
                const float scale=float(sycl::bit_cast<sycl::half>(bits));
                const unsigned raw=weights[offset+2+column%32];
                const int code=raw<128?int(raw):int(raw)-256;
                const float weight=scale*float(code);
                for(int j=0;j<Tile;++j)if(first+j<tokens)
                    acc[j]=sycl::fma(weight,input[size_t(first+j)*k+column],acc[j]);
            }
            for(int j=0;j<Tile;++j) {
                // Each token's validity is uniform across this whole subgroup.
                const float result=sum32(item.get_sub_group(),acc[j]);
                if(lane==0 && first+j<tokens)output[size_t(first+j)*m+row]=result;
            }
        });
}
int token_tile_requested() {
    static const int tile=[] {const char* v=std::getenv("STRATA_HC_Q8_TOKEN_TILE");
        if(!v || std::strcmp(v,"0")==0)return 0;
        if(std::strcmp(v,"2")==0)return 2;
        if(std::strcmp(v,"4")==0)return 4;
        return -1;}();
    return tile;
}
}

bool hc_q8_0_project_f32(const uint8_t * weights, size_t weight_bytes,
        const float * input, size_t input_floats, float * output, size_t output_floats,
        int k, int m, int tokens, void * stream) {
    if (!weights || !input || !output || !hc_projection_detail::shape_ok(k,m,tokens,input_floats,output_floats) ||
        weight_bytes < size_t(m)*(k/32)*34) { return false; }
    auto * queue = strata::q_of(stream);
    if (!queue->has_property<sycl::property::queue::in_order>()) { return false; }
    const int tile=token_tile_requested();
    if(tile<0)return false;
    if(tile && tokens>=2)return hc_q8_0_project_f32_token_reuse(weights,weight_bytes,input,input_floats,output,output_floats,k,m,tokens,tile,stream);
    launch<true>(*queue,weights,input,output,k,m,tokens);
    return true;
}

bool hc_f32_project_f32(const float * weights, size_t weight_floats,
        const float * input, size_t input_floats, float * output, size_t output_floats,
        int k, int m, int tokens, void * stream) {
    if (!weights || !input || !output || !hc_projection_detail::shape_ok(k,m,tokens,input_floats,output_floats) ||
        weight_floats < size_t(m)*k) { return false; }
    auto * queue = strata::q_of(stream);
    if (!queue->has_property<sycl::property::queue::in_order>()) { return false; }
    launch<false>(*queue,weights,input,output,k,m,tokens);
    return true;
}

bool hc_q8_0_project_f32_token_reuse(const uint8_t* weights,size_t weight_bytes,
        const float* input,size_t input_floats,float* output,size_t output_floats,
        int k,int m,int tokens,int tile,void* stream) {
    // Explicit operator-only T512 entry; production descriptors/caller remain <=8.
    if(!weights || !input || !output || k<=0 || k>10240 || k%32 || m<=0 || m>10240 ||
       tokens<1 || tokens>512 || input_floats<size_t(k)*tokens || output_floats<size_t(m)*tokens ||
       weight_bytes<size_t(m)*(k/32)*34 || (tile!=2 && tile!=4))return false;
    auto* queue=strata::q_of(stream);
    if(!queue->has_property<sycl::property::queue::in_order>())return false;
    if(tile==2)launch_q8_token_reuse<2>(*queue,weights,input,output,k,m,tokens);
    else launch_q8_token_reuse<4>(*queue,weights,input,output,k,m,tokens);
    return true;
}
bool hc_f32_project_f32_prefill(const float* weights,size_t weight_floats,
        const float* input,size_t input_floats,float* output,size_t output_floats,
        int k,int m,int tokens,void* stream) {
    if(!weights || !input || !output || k<=0 || k>10240 || k%32 || m<=0 || m>10240 ||
       tokens<1 || tokens>512 || input_floats<size_t(k)*tokens || output_floats<size_t(m)*tokens ||
       weight_floats<size_t(m)*k)return false;
    auto* queue=strata::q_of(stream);
    if(!queue->has_property<sycl::property::queue::in_order>())return false;
    launch<false>(*queue,weights,input,output,k,m,tokens);
    return true;
}

}
