#pragma once
#include <cstdint>
#include <vector>

namespace b70ep {

std::vector<int64_t> m0_setup(int spin);
void m0_prep(int sample);
void m0_submit_card1();
void m0_submit_card0(int mirror);
std::vector<int64_t> m0_check(int sample);
void m0_card1_capture_begin();
void m0_card1_capture_end();
void m0_card1_replay();
void m0_latency();
void m0_teardown();

void ep_init(int max_t, int max_k, int max_topk, int nlayers);
void ep_register0(int layer, const uint8_t* gate, int gate_type, const uint8_t* up, int up_type, const uint8_t* down,
                  int down_type, int e, int intermediate, int hidden);
void ep_register1(int layer, const uint8_t* gate, const uint8_t* up, const uint8_t* down, int e);
void ep_register_map(int layer, const int32_t* map0, const int32_t* map1, int nexp);
void ep_moe(const uint16_t* x, const float* w, const int32_t* ids, float* y, int T, int topk, int K, int layer);
void ep_register_shared(int layer, const uint8_t* sg, int sgt, const uint8_t* su, int sut, const uint8_t* sd, int sdt,
                        const void* sgi, int sgi_dt);
void ep_moe_sh(const uint16_t* x, const float* w, const int32_t* ids, float* y, int T, int topk, int K, int layer,
               bool bf16_out = false);
void ep_moe_route(const uint16_t* x, const uint16_t* logits, uint16_t* y, int T, int E, int topk, int K, int layer,
                  int renorm);
void ep_moe_begin(const uint16_t* x, const float* w, const int32_t* ids, int T, int topk, int K, int layer);
void ep_moe_end(float* y, int T, int K, int layer);
void ep_capture_begin(int key);
void ep_capture_end(int key);
void ep_replay_prepare(int key);
void ep_replay_submit(int key);
void ep_replay(int key);
std::vector<int64_t> ep_read_errors();
void ep_reserve();
// F1: 0 before ep_init, else the peer-pointer path (1 IPC, 2 same-context, 3 USM).
int ep_transport_path();

}  // namespace b70ep
