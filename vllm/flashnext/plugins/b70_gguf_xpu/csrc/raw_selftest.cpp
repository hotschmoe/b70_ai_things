// Host check: b70_load_raw + b70_decode_raw vs b70_decode32, bit for bit.
#include "grouped_raw.hpp"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

static int fail = 0;

template <int TYPE>
void check(int K, uint32_t seed) {
    const int rb = b70_row_bytes(TYPE, K);
    if (rb <= 0) {
        std::printf("SKIP type %d K %d\n", TYPE, K);
        return;
    }
    std::vector<uint8_t> row((size_t)rb);
    uint32_t s = seed;
    for (size_t i = 0; i < row.size(); ++i) {
        s = s * 1664525u + 1013904223u;
        row[i] = (uint8_t)(s >> 24);
    }
    // Finite fp16 scales (1.0) so both paths stay off the inf path.
    auto put16 = [&](int off) {
        row[(size_t)off] = 0x00;
        row[(size_t)off + 1] = 0x3c;
    };
    if (TYPE == B70_Q6_K) {
        for (int b = 0; b < K / 256; ++b) put16(b * 210 + 208);
    } else if (TYPE == B70_Q4_K) {
        for (int b = 0; b < K / 256; ++b) { put16(b * 144); put16(b * 144 + 2); }
    } else if (TYPE == B70_Q5_K) {
        for (int b = 0; b < K / 256; ++b) { put16(b * 176); put16(b * 176 + 2); }
    } else if (TYPE == B70_IQ4_XS) {
        for (int b = 0; b < K / 256; ++b) put16(b * 136);
    } else if (TYPE == B70_Q2_0) {
        for (int b = 0; b < K / 64; ++b) put16(b * 18);
    } else if (TYPE == B70_IQ4_NL) {
        for (int b = 0; b < K / 32; ++b) put16(b * 18);
    } else {
        int be = 0, bb = 0;
        b70_type_info(TYPE, be, bb);
        for (int b = 0; b < K / be; ++b) put16(b * bb);
    }
    const int groups = K / 32;
    int bad = 0;
    for (int gi = 0; gi < groups; ++gi) {
        float a[32], b[32];
        b70_decode32<TYPE>(row.data(), gi, a);
        b70_raw_decode_gi<TYPE>(row.data(), gi, b);
        for (int j = 0; j < 32; ++j) {
            uint32_t ba, bbv;
            memcpy(&ba, &a[j], 4);
            memcpy(&bbv, &b[j], 4);
            if (ba != bbv) {
                if (bad < 4)
                    std::printf("MIS type %d K %d gi %d j %d a %a b %a\n", TYPE, K, gi, j, a[j], b[j]);
                ++bad;
            }
        }
    }
    // Flat matrix index: row 1 of a 2-row matrix must match row pointer + rb.
    std::vector<uint8_t> mat(2 * (size_t)rb);
    memcpy(mat.data(), row.data(), row.size());
    memcpy(mat.data() + rb, row.data(), row.size());
    for (int gi = 0; gi < groups; gi += groups / 3 + 1) {
        const int64_t flat = (int64_t)1 * K + (int64_t)gi * 32;
        float a[32], b[32];
        b70_decode32<TYPE>(mat.data() + rb, gi, a);
        const int ib = (int)((flat & 255) >> 5);
        b70_decode_raw<TYPE>(b70_load_raw<TYPE>(mat.data(), flat >> 8, ib), ib, b);
        for (int j = 0; j < 32; ++j) {
            uint32_t ba, bbv;
            memcpy(&ba, &a[j], 4);
            memcpy(&bbv, &b[j], 4);
            if (ba != bbv) {
                if (bad < 8)
                    std::printf("MIS flat type %d K %d gi %d j %d\n", TYPE, K, gi, j);
                ++bad;
            }
        }
    }
    std::printf("%s type %d K %d groups %d bad %d\n", bad ? "FAIL" : "OK", TYPE, K, groups, bad);
    fail += bad ? 1 : 0;
}

int main() {
    const int types[] = {12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 42};
    for (int t : types) {
        switch (t) {
        case 12: check<12>(256, 1); check<12>(2560, 2); break;
        case 13: check<13>(256, 3); check<13>(2560, 4); break;
        case 14: check<14>(256, 5); check<14>(2560, 6); break;
        case 16: check<16>(256, 7); check<16>(2560, 8); break;
        case 17: check<17>(256, 9); check<17>(2560, 10); break;
        case 18: check<18>(256, 11); check<18>(2560, 12); break;
        case 20: check<20>(32, 13); check<20>(640, 14); check<20>(2560, 15); break;
        case 21: check<21>(256, 16); check<21>(2560, 17); break;
        case 22: check<22>(256, 18); check<22>(2560, 19); break;
        case 23: check<23>(256, 20); check<23>(2560, 21); break;
        case 42: check<42>(64, 22); check<42>(640, 23); check<42>(2560, 24); break;
        }
    }
    std::printf("VERDICT fail_types=%d\n", fail);
    return fail ? 1 : 0;
}
