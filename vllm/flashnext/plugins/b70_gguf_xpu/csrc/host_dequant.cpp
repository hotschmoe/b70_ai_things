#include "decode.hpp"
extern "C" {
void b70_dequant_f32_cpu(int type, const uint8_t* q, long long rows, long long cols, float* out) {
    b70_dequant_f32_rows(type, q, rows, cols, out);
}
int b70_row_bytes_cpu(int type, int k) { return b70_row_bytes(type, k); }
}
