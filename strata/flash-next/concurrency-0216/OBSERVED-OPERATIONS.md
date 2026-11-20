# Ranked recorded operation service

These are per-card operation totals in scenario-bounded profiled intervals, not proven critical-path
costs. They can overlap other work, include edge windows and profiler overhead, and omit work
not recorded by unitrace. Exact names, shapes, p50/p95 and device/thread attribution are in the
per-scenario CSV/JSON. B5 may omit graphs captured during paused preparation.
Host API duration includes waits and profiler overhead; it is not CPU time or a separate
additive cost, and can overlap device execution.

## B1

### Card 0

| Rank | Operation | Launches | Recorded service ms | p50 / p95 us |
| ---: | --- | ---: | ---: | ---: |
| 1 | gr_q8::launch_down<4> [SIMD16 {120; 1; 1} {256; 1; 1}] | 7954 | 137.36 | 17.08 / 18.75 |
| 2 | kv_append_q8_steps [SIMD32 {2; 4; 8} {64; 1; 1}] | 1031 | 111.04 | 107.81 / 111.77 |
| 3 | gr_q8::launch_up<4> [SIMD16 {160; 1; 1} {320; 1; 1}] | 7954 | 102.97 | 12.81 / 14.27 |
| 4 | launch_kreg<KregQ5K, 4, 4, 128> [SIMD16 {62080; 1; 1} {80; 1; 1}] | 82 | 77.03 | 940.26 / 944.47 |
| 5 | launch_kreg<KregQ6K, 4, 2, 128> [SIMD16 {1280; 1; 1} {192; 1; 1}] | 2378 | 70.24 | 29.38 / 30.31 |
| 6 | launch_down_q2_word_expand [SIMD32 {80; 40; 1} {256; 1; 1}] | 2460 | 58.49 | 23.85 / 30.31 |
| 7 | launch_night_kreg2<NightIQ3S, 4, 8, 256> [SIMD16 {768; 1; 1} {80; 1; 1}] | 820 | 55.00 | 67.19 / 69.69 |
| 8 | launch_kreg<KregQ6K, 4, 2, 128> [SIMD16 {5120; 1; 1} {80; 1; 1}] | 1148 | 52.22 | 45.52 / 46.35 |
| 9 | launch_gu_port<16, 8, true> [SIMD32 {40; 40; 1} {256; 1; 1}] | 738 | 52.05 | 70.94 / 92.77 |
| 10 | launch_gu_row_reuse<21> [SIMD16 {20; 40; 1} {256; 1; 1}] | 1066 | 51.39 | 44.79 / 62.99 |

### Card 1

| Rank | Operation | Launches | Recorded service ms | p50 / p95 us |
| ---: | --- | ---: | ---: | ---: |
| 1 | ep_wait [SIMD32 {1; 1; 1} {1; 1; 1}] | 3936 | 1451.77 | 308.02 / 556.64 |
| 2 | launch_gu_row_reuse<21> [SIMD16 {20; 40; 1} {256; 1; 1}] | 1066 | 59.43 | 57.81 / 77.50 |
| 3 | launch_down_q2_word_expand [SIMD32 {80; 40; 1} {256; 1; 1}] | 2460 | 59.29 | 24.17 / 30.42 |
| 4 | launch_down_rows_k<20, 8, 2> [SIMD16 {40; 40; 1} {256; 1; 1}] | 1476 | 56.03 | 37.60 / 52.60 |
| 5 | launch_gu_port<17, 8, true> [SIMD32 {40; 40; 1} {256; 1; 1}] | 820 | 52.54 | 64.58 / 82.09 |
| 6 | launch_gu_port<16, 8, true> [SIMD32 {40; 40; 1} {256; 1; 1}] | 738 | 49.19 | 65.68 / 87.23 |
| 7 | launch_gu_row_reuse<22> [SIMD16 {20; 40; 1} {256; 1; 1}] | 820 | 41.24 | 50.36 / 65.21 |
| 8 | launch_gu_row_reuse<18> [SIMD16 {20; 40; 1} {256; 1; 1}] | 492 | 28.22 | 58.75 / 76.14 |
| 9 | ep_send_psum [SIMD32 {40; 1; 1} {64; 1; 1}] | 3936 | 28.03 | 7.08 / 7.81 |
| 10 | native_expert_grouped [SIMD32 {100; 1; 1} {256; 1; 1}] | 3936 | 10.76 | 2.71 / 3.12 |

### Recorded host Level Zero calls

| Rank | API | Calls | Summed duration ms | p50 / p95 us |
| ---: | --- | ---: | ---: | ---: |
| 1 | zeCommandListHostSynchronize | 41784 | 2282.54 | 2.06 / 2.98 |
| 2 | zeCommandListImmediateAppendCommandListsExp | 489 | 10.95 | 11.88 / 48.47 |
| 3 | zeEventHostSynchronize | 500 | 1.63 | 0.94 / 1.97 |
| 4 | zeCommandListCreate | 2 | 1.20 | 598.59 / 659.30 |
| 5 | zeCommandListAppendLaunchKernel | 129 | 0.60 | 1.97 / 3.49 |
| 6 | zeCommandListAppendBarrier | 82 | 0.51 | 5.86 / 8.26 |
| 7 | zeModuleCreate | 1 | 0.29 | 291.84 / 291.84 |
| 8 | zeCommandListAppendMemoryCopy | 12 | 0.26 | 11.89 / 62.46 |
| 9 | zeKernelSetArgumentValue | 611 | 0.14 | 0.10 / 0.83 |
| 10 | zeKernelSetGroupSize | 129 | 0.01 | 0.04 / 0.18 |

## B2

### Card 0

| Rank | Operation | Launches | Recorded service ms | p50 / p95 us |
| ---: | --- | ---: | ---: | ---: |
| 1 | wait_flag_ge [SIMD32 {1; 1; 1} {1; 1; 1}] | 136 | 335.81 | 2.08 / 25286.69 |
| 2 | gr_q8::launch_down<4> [SIMD16 {120; 1; 1} {256; 1; 1}] | 13095 | 236.75 | 17.81 / 20.62 |
| 3 | gr_q8::launch_up<4> [SIMD16 {160; 1; 1} {320; 1; 1}] | 13095 | 178.53 | 13.44 / 15.31 |
| 4 | QsaVectorCodes[SIMD32 {33; 2; 1} {256; 1; 1}] [SIMD32 {33; 2; 1} {256; 1; 1}] | 6504 | 157.13 | 25.00 / 29.69 |
| 5 | gdn_step_norm_multi [SIMD32 {48; 4; 1} {32; 4; 1}] | 9756 | 149.16 | 15.73 / 16.77 |
| 6 | launch_kreg<KregQ5K, 4, 4, 128> [SIMD16 {62080; 1; 1} {80; 1; 1}] | 135 | 127.49 | 941.56 / 958.20 |
| 7 | gdn_step_norm_groups [SIMD32 {48; 2; 1} {128; 4; 1}] | 4860 | 122.78 | 25.10 / 26.98 |
| 8 | launch_kreg<KregQ6K, 4, 2, 128> [SIMD16 {1280; 1; 1} {192; 1; 1}] | 3915 | 120.31 | 30.62 / 34.17 |
| 9 | launch_down_q2_word_expand [SIMD32 {80; 40; 1} {256; 1; 1}] | 4050 | 97.42 | 23.96 / 31.46 |
| 10 | launch_night_kreg2<NightIQ3S, 4, 8, 256> [SIMD16 {768; 1; 1} {80; 1; 1}] | 1350 | 91.99 | 66.98 / 80.73 |

### Card 1

| Rank | Operation | Launches | Recorded service ms | p50 / p95 us |
| ---: | --- | ---: | ---: | ---: |
| 1 | ep_wait [SIMD32 {1; 1; 1} {1; 1; 1}] | 6528 | 3123.47 | 330.31 / 677.57 |
| 2 | launch_down_q2_word_expand [SIMD32 {80; 40; 1} {256; 1; 1}] | 4050 | 99.46 | 24.27 / 32.97 |
| 3 | launch_gu_row_reuse<21> [SIMD16 {20; 40; 1} {256; 1; 1}] | 1755 | 98.93 | 58.12 / 78.99 |
| 4 | launch_down_rows_k<20, 8, 2> [SIMD16 {40; 40; 1} {256; 1; 1}] | 2430 | 92.12 | 37.50 / 53.44 |
| 5 | launch_gu_port<17, 8, true> [SIMD32 {40; 40; 1} {256; 1; 1}] | 1350 | 90.52 | 65.94 / 89.79 |
| 6 | launch_gu_port<16, 8, true> [SIMD32 {40; 40; 1} {256; 1; 1}] | 1215 | 84.33 | 69.38 / 94.78 |
| 7 | launch_gu_row_reuse<22> [SIMD16 {20; 40; 1} {256; 1; 1}] | 1350 | 68.71 | 50.94 / 67.39 |
| 8 | launch_gu_row_reuse<18> [SIMD16 {20; 40; 1} {256; 1; 1}] | 810 | 47.73 | 60.10 / 80.32 |
| 9 | ep_send_psum [SIMD32 {40; 1; 1} {64; 1; 1}] | 6480 | 47.28 | 7.19 / 8.44 |
| 10 | native_expert_grouped [SIMD32 {100; 1; 1} {256; 1; 1}] | 6480 | 17.89 | 2.71 / 3.54 |

### Recorded host Level Zero calls

| Rank | API | Calls | Summed duration ms | p50 / p95 us |
| ---: | --- | ---: | ---: | ---: |
| 1 | zeCommandListHostSynchronize | 1356 | 4405.66 | 3.19 / 28332.53 |
| 2 | zeCommandListImmediateAppendCommandListsExp | 812 | 25.18 | 16.02 / 57.76 |
| 3 | zeCommandListAppendLaunchKernel | 4451 | 13.15 | 1.64 / 2.39 |
| 4 | zeCommandListCreate | 13 | 9.06 | 721.14 / 843.63 |
| 5 | zeCommandListAppendMemoryCopy | 345 | 6.00 | 8.44 / 35.70 |
| 6 | zeKernelSetArgumentValue | 30368 | 3.04 | 0.06 / 0.30 |
| 7 | zeEventHostSynchronize | 1081 | 2.64 | 1.43 / 4.80 |
| 8 | zeModuleCreate | 3 | 0.73 | 194.46 / 351.76 |
| 9 | zeContextMakeMemoryResident | 2 | 0.40 | 197.84 / 235.59 |
| 10 | zeKernelSetGroupSize | 4463 | 0.25 | 0.04 / 0.12 |

## B3

### Card 0

| Rank | Operation | Launches | Recorded service ms | p50 / p95 us |
| ---: | --- | ---: | ---: | ---: |
| 1 | wait_flag_ge [SIMD32 {1; 1; 1} {1; 1; 1}] | 302 | 906.79 | 2.08 / 25575.03 |
| 2 | gr_q8::launch_down<8> [SIMD16 {120; 1; 1} {256; 1; 1}] | 27451 | 884.35 | 31.98 / 36.04 |
| 3 | QsaVectorCodes[SIMD32 {33; 2; 1} {256; 1; 1}] [SIMD32 {33; 2; 1} {256; 1; 1}] | 28248 | 749.09 | 26.35 / 30.10 |
| 4 | gdn_step_norm_multi [SIMD32 {48; 4; 1} {32; 4; 1}] | 42372 | 649.75 | 15.62 / 16.56 |
| 5 | qsa_topk_wide_scan_k[SIMD32 {1; 1; 1} {1024; 1; 1}] [SIMD32 {1; 1; 1} {1024; 1; 1}] | 28248 | 542.44 | 18.85 / 22.50 |
| 6 | gr_q8::launch_up<8> [SIMD16 {160; 1; 1} {320; 1; 1}] | 27451 | 513.80 | 18.85 / 20.94 |
| 7 | gdn_step_norm_groups [SIMD32 {48; 4; 1} {128; 4; 1}] | 10188 | 491.89 | 48.12 / 50.83 |
| 8 | launch_kreg<KregQ5K, 8, 2, 128> [SIMD16 {1280; 1; 1} {192; 1; 1}] | 1415 | 374.02 | 262.71 / 280.69 |
| 9 | launch_down_q2_word_expand [SIMD32 {80; 80; 1} {256; 1; 1}] | 8490 | 371.40 | 43.23 / 54.90 |
| 10 | launch_kreg<KregQ5K, 8, 8, 256> [SIMD16 {31040; 1; 1} {80; 1; 1}] | 283 | 352.94 | 1255.83 / 1260.92 |

### Card 1

| Rank | Operation | Launches | Recorded service ms | p50 / p95 us |
| ---: | --- | ---: | ---: | ---: |
| 1 | ep_wait [SIMD32 {1; 1; 1} {1; 1; 1}] | 14496 | 11470.10 | 491.88 / 1509.38 |
| 2 | launch_down_q2_word_expand [SIMD32 {80; 80; 1} {256; 1; 1}] | 8490 | 384.13 | 45.00 / 56.88 |
| 3 | launch_gu_row_reuse<21> [SIMD16 {20; 80; 1} {256; 1; 1}] | 3679 | 358.47 | 98.02 / 130.45 |
| 4 | launch_down_rows_k<20, 8, 2> [SIMD16 {40; 80; 1} {256; 1; 1}] | 5094 | 329.01 | 63.85 / 87.33 |
| 5 | launch_gu_port<16, 8, true> [SIMD32 {40; 80; 1} {256; 1; 1}] | 2547 | 324.26 | 125.73 / 163.90 |
| 6 | launch_gu_port<17, 8, true> [SIMD32 {40; 80; 1} {256; 1; 1}] | 2830 | 323.40 | 112.39 / 146.98 |
| 7 | launch_gu_row_reuse<22> [SIMD16 {20; 80; 1} {256; 1; 1}] | 2830 | 234.78 | 82.71 / 106.20 |
| 8 | launch_gu_row_reuse<18> [SIMD16 {20; 80; 1} {256; 1; 1}] | 1698 | 165.02 | 98.96 / 120.52 |
| 9 | ep_send_psum [SIMD32 {80; 1; 1} {64; 1; 1}] | 13584 | 148.33 | 10.73 / 11.88 |
| 10 | native_expert_grouped [SIMD32 {200; 1; 1} {256; 1; 1}] | 13584 | 39.99 | 2.92 / 3.54 |

### Recorded host Level Zero calls

| Rank | API | Calls | Summed duration ms | p50 / p95 us |
| ---: | --- | ---: | ---: | ---: |
| 1 | zeCommandListHostSynchronize | 3971 | 16581.02 | 2.94 / 47372.78 |
| 2 | zeCommandListImmediateAppendCommandListsExp | 2657 | 68.32 | 12.64 / 63.39 |
| 3 | zeCommandListAppendMemoryCopy | 1302 | 56.31 | 6.66 / 38.48 |
| 4 | zeModuleCreate | 25 | 16.70 | 526.02 / 1717.04 |
| 5 | zeCommandListAppendLaunchKernel | 4842 | 13.85 | 1.65 / 2.25 |
| 6 | zeEventHostSynchronize | 3887 | 12.01 | 1.10 / 4.78 |
| 7 | zeCommandListCreate | 6 | 5.08 | 856.06 / 1103.03 |
| 8 | zeKernelSetArgumentValue | 33996 | 3.09 | 0.07 / 0.25 |
| 9 | zeKernelSetGroupSize | 4854 | 0.27 | 0.04 / 0.10 |
| 10 | zeKernelCreate | 25 | 0.16 | 2.75 / 25.23 |

## B4

### Card 0

| Rank | Operation | Launches | Recorded service ms | p50 / p95 us |
| ---: | --- | ---: | ---: | ---: |
| 1 | prefill::kv_append [SIMD32 {12288; 2; 8} {64; 1; 1}] | 48 | 18477.54 | 385027.45 / 386538.53 |
| 2 | prefill::kv_append [SIMD32 {7503; 2; 8} {64; 1; 1}] | 52 | 12260.63 | 235663.96 / 236638.97 |
| 3 | qsa_union_attn[SIMD16 {256; 2; 1} {192; 1; 1}] [SIMD16 {256; 2; 1} {192; 1; 1}] | 424 | 4888.36 | 12625.73 / 15170.41 |
| 4 | prefill::gdn_recurrence_variant [SIMD32 {192; 1; 1} {32; 4; 1}] | 288 | 3989.11 | 14098.23 / 16613.61 |
| 5 | gemm_f16f16f32_nocopy_tn_32x64_8x4[SIMD16 {32; 1; 1} {128; 4; 1}] [SIMD16 {32; 1; 1} {128; 4; 1}] | 3654 | 3293.06 | 242.40 / 4787.95 |
| 6 | ep_push_signal [SIMD32 {4096; 1; 1} {256; 1; 1}] | 384 | 1802.08 | 4732.92 / 5839.46 |
| 7 | prefill::kv_append [SIMD32 {11520; 2; 8} {64; 1; 1}] | 4 | 1443.14 | 360726.67 / 361903.58 |
| 8 | wait_flag_ge [SIMD32 {1; 1; 1} {1; 1; 1}] | 583 | 1363.99 | 2.08 / 25421.65 |
| 9 | gemm_bf16bf16f32_nocopy_tn_32x64_8x4[SIMD16 {32; 1; 1} {128; 4; 1}] [SIMD16 {32; 1; 1} {128; 4; 1}] | 1645 | 1294.34 | 758.64 / 1413.10 |
| 10 | prefill::gr_write_norm_rs [SIMD32 {49152; 1; 1} {256; 1; 1}] | 373 | 1113.58 | 2983.96 / 3006.60 |

### Card 1

| Rank | Operation | Launches | Recorded service ms | p50 / p95 us |
| ---: | --- | ---: | ---: | ---: |
| 1 | ep_wait [SIMD32 {1; 1; 1} {1; 1; 1}] | 27984 | 14475.13 | 349.06 / 1046.96 |
| 2 | ep_prefill_reduce_h_kernel<true>[SIMD32 {64; 1; 1} {160; 1; 1}] [SIMD32 {64; 1; 1} {160; 1; 1}] | 384 | 1915.99 | 4840.52 / 6653.90 |
| 3 | gemm_f16f16f32_nocopy_tn_32x32_4x4x2[SIMD16 {10; 3; 1} {64; 4; 2}] [SIMD16 {10; 3; 1} {64; 4; 2}] | 7074 | 289.67 | 38.96 / 48.85 |
| 4 | gemm_f16f16f32_nocopy_tn_32x32_4x2x4[SIMD16 {10; 3; 1} {64; 2; 4}] [SIMD16 {10; 3; 1} {64; 2; 4}] | 10840 | 274.40 | 24.58 / 29.48 |
| 5 | gemm_f16f16f32_nocopy_tn_32x64_8x2x2[SIMD16 {5; 4; 1} {128; 2; 2}] [SIMD16 {5; 4; 1} {128; 2; 2}] | 4166 | 259.11 | 58.54 / 76.14 |
| 6 | gemm_f16f16f32_nocopy_tn_32x32_4x4x2[SIMD16 {10; 2; 1} {64; 4; 2}] [SIMD16 {10; 2; 1} {64; 4; 2}] | 6393 | 245.60 | 36.25 / 46.98 |
| 7 | launch_gu_row_reuse<21> [SIMD16 {20; 60; 1} {256; 1; 1}] | 2483 | 217.31 | 84.27 / 115.52 |
| 8 | launch_down_q2_word_expand [SIMD32 {80; 60; 1} {256; 1; 1}] | 5730 | 213.49 | 36.98 / 46.72 |
| 9 | launch_gu_row_reuse<21> [SIMD16 {20; 40; 1} {256; 1; 1}] | 3341 | 207.45 | 61.67 / 81.56 |
| 10 | gemm_f16f16f32_nocopy_tn_32x64_8x4[SIMD16 {32; 1; 1} {128; 4; 1}] [SIMD16 {32; 1; 1} {128; 4; 1}] | 2383 | 207.30 | 57.50 / 187.98 |

### Recorded host Level Zero calls

| Rank | API | Calls | Summed duration ms | p50 / p95 us |
| ---: | --- | ---: | ---: | ---: |
| 1 | zeCommandListAppendMemoryCopy | 5789 | 51400.42 | 34.79 / 34365.52 |
| 2 | zeCommandListHostSynchronize | 32269 | 24479.32 | 2.06 / 1209.29 |
| 3 | zeCommandListAppendLaunchKernelWithArguments | 374076 | 2566.01 | 6.38 / 8.50 |
| 4 | zeCommandListImmediateAppendCommandListsExp | 3390 | 99.54 | 16.04 / 61.78 |
| 5 | zeMemGetAllocProperties | 859404 | 76.88 | 0.07 / 0.16 |
| 6 | zeCommandListAppendLaunchKernel | 12266 | 36.12 | 1.65 / 2.29 |
| 7 | zeKernelSetGroupSize | 386342 | 35.18 | 0.08 / 0.19 |
| 8 | zeModuleCreate | 62 | 34.42 | 454.46 / 1631.88 |
| 9 | zeEventHostSynchronize | 5439 | 23.40 | 1.57 / 10.77 |
| 10 | zeCommandListCreate | 28 | 17.26 | 633.85 / 771.79 |

## B5

### Card 0

| Rank | Operation | Launches | Recorded service ms | p50 / p95 us |
| ---: | --- | ---: | ---: | ---: |
| 1 | zeCommandListAppendMemoryCopy | 7 | 184.08 | 26254.90 / 26524.56 |
| 2 | zeCommandListAppendMemoryCopy | 36 | 134.97 | 3739.04 / 3794.71 |
| 3 | zeCommandListAppendMemoryCopy | 24 | 53.69 | 2228.70 / 2301.14 |
| 4 | qsa_topk_wide_scan_k[SIMD32 {1; 1; 1} {1024; 1; 1}] [SIMD32 {1; 1; 1} {1024; 1; 1}] | 576 | 38.65 | 65.83 / 83.98 |
| 5 | zeCommandListAppendMemoryCopy | 2 | 32.48 | 16240.83 / 16335.90 |
| 6 | wait_flag_ge [SIMD32 {1; 1; 1} {1; 1; 1}] | 12 | 26.27 | 2.19 / 11811.53 |
| 7 | gr_q8::launch_down<4> [SIMD16 {120; 1; 1} {256; 1; 1}] | 1164 | 21.28 | 18.02 / 20.73 |
| 8 | qsa_scores_transposed_k<1>[SIMD32 {256; 1; 1} {256; 1; 1}] [SIMD32 {256; 1; 1} {256; 1; 1}] | 576 | 17.61 | 36.35 / 40.94 |
| 9 | gr_q8::launch_up<4> [SIMD16 {160; 1; 1} {320; 1; 1}] | 1164 | 16.19 | 13.54 / 15.62 |
| 10 | QsaVectorCodes[SIMD32 {33; 2; 1} {256; 1; 1}] [SIMD32 {33; 2; 1} {256; 1; 1}] | 576 | 16.04 | 27.55 / 33.25 |

### Card 1

| Rank | Operation | Launches | Recorded service ms | p50 / p95 us |
| ---: | --- | ---: | ---: | ---: |
| 1 | ep_wait [SIMD32 {1; 1; 1} {1; 1; 1}] | 576 | 336.54 | 320.73 / 1237.24 |
| 2 | launch_gu_row_reuse<21> [SIMD16 {20; 40; 1} {256; 1; 1}] | 156 | 11.22 | 69.58 / 92.32 |
| 3 | launch_down_q2_word_expand [SIMD32 {80; 40; 1} {256; 1; 1}] | 360 | 10.75 | 29.48 / 38.96 |
| 4 | launch_down_rows_k<20, 8, 2> [SIMD16 {40; 40; 1} {256; 1; 1}] | 216 | 10.05 | 45.99 / 58.70 |
| 5 | launch_gu_port<16, 8, true> [SIMD32 {40; 40; 1} {256; 1; 1}] | 108 | 9.47 | 91.98 / 110.21 |
| 6 | launch_gu_port<17, 8, true> [SIMD32 {40; 40; 1} {256; 1; 1}] | 120 | 9.34 | 78.65 / 99.91 |
| 7 | launch_gu_row_reuse<22> [SIMD16 {20; 40; 1} {256; 1; 1}] | 120 | 7.02 | 55.73 / 70.48 |
| 8 | launch_gu_row_reuse<18> [SIMD16 {20; 40; 1} {256; 1; 1}] | 72 | 5.20 | 68.07 / 92.02 |
| 9 | ep_send_psum [SIMD32 {40; 1; 1} {64; 1; 1}] | 576 | 4.28 | 7.29 / 8.75 |
| 10 | native_expert_grouped [SIMD32 {100; 1; 1} {256; 1; 1}] | 576 | 1.66 | 2.81 / 3.75 |

### Recorded host Level Zero calls

| Rank | API | Calls | Summed duration ms | p50 / p95 us |
| ---: | --- | ---: | ---: | ---: |
| 1 | zeCommandListHostSynchronize | 1075 | 1020.54 | 2.07 / 2242.89 |
| 2 | zeCommandListAppendMemoryCopy | 948 | 456.48 | 11.22 / 2520.85 |
| 3 | zeCommandListAppendLaunchKernel | 2238 | 6.69 | 1.63 / 2.19 |
| 4 | zeEventHostSynchronize | 773 | 5.95 | 3.95 / 37.81 |
| 5 | zeCommandListImmediateAppendCommandListsExp | 162 | 3.81 | 13.24 / 55.18 |
| 6 | zeCommandListCreate | 3 | 2.11 | 732.45 / 778.44 |
| 7 | zeMemGetAllocProperties | 2612 | 1.36 | 0.46 / 1.28 |
| 8 | zeKernelSetArgumentValue | 15700 | 1.34 | 0.06 / 0.22 |
| 9 | zeCommandListAppendLaunchKernelWithArguments | 60 | 0.99 | 12.43 / 23.52 |
| 10 | zeKernelSetGroupSize | 2298 | 0.15 | 0.04 / 0.27 |

## B6

### Card 0

| Rank | Operation | Launches | Recorded service ms | p50 / p95 us |
| ---: | --- | ---: | ---: | ---: |
| 1 | prefill::kv_append [SIMD32 {12288; 2; 8} {64; 1; 1}] | 24 | 9268.31 | 386177.08 / 386409.25 |
| 2 | prefill::kv_append [SIMD32 {8145; 2; 8} {64; 1; 1}] | 13 | 3327.98 | 255513.85 / 257558.40 |
| 3 | qsa_union_attn[SIMD16 {256; 2; 1} {192; 1; 1}] [SIMD16 {256; 2; 1} {192; 1; 1}] | 178 | 2803.28 | 17005.31 / 20561.54 |
| 4 | prefill::gdn_recurrence_variant [SIMD32 {192; 1; 1} {32; 4; 1}] | 108 | 1622.03 | 16511.09 / 16676.77 |
| 5 | gemm_f16f16f32_nocopy_tn_32x64_8x4[SIMD16 {32; 1; 1} {128; 4; 1}] [SIMD16 {32; 1; 1} {128; 4; 1}] | 1822 | 1621.40 | 250.26 / 4790.96 |
| 6 | ep_push_signal [SIMD32 {4096; 1; 1} {256; 1; 1}] | 144 | 742.91 | 5776.09 / 5855.30 |
| 7 | prefill::kv_append [SIMD32 {11520; 2; 8} {64; 1; 1}] | 2 | 725.83 | 362915.94 / 364294.25 |
| 8 | gemm_bf16bf16f32_nocopy_tn_32x64_8x4[SIMD16 {32; 1; 1} {128; 4; 1}] [SIMD16 {32; 1; 1} {128; 4; 1}] | 765 | 653.23 | 929.38 / 1413.73 |
| 9 | prefill::gr_write_norm_rs [SIMD32 {49152; 1; 1} {256; 1; 1}] | 188 | 560.89 | 2981.14 / 2996.56 |
| 10 | prefill::gr_mix_r [SIMD32 {122880; 1; 1} {256; 1; 1}] | 192 | 469.92 | 2442.29 / 2480.82 |

### Card 1

| Rank | Operation | Launches | Recorded service ms | p50 / p95 us |
| ---: | --- | ---: | ---: | ---: |
| 1 | ep_prefill_reduce_h_kernel<true>[SIMD32 {64; 1; 1} {160; 1; 1}] [SIMD32 {64; 1; 1} {160; 1; 1}] | 144 | 847.78 | 6306.09 / 7275.04 |
| 2 | gemm_f16f16f32_nocopy_tn_32x64_8x4[SIMD16 {32; 1; 1} {128; 4; 1}] [SIMD16 {32; 1; 1} {128; 4; 1}] | 2017 | 245.81 | 86.04 / 297.44 |
| 3 | gemm_f16f16f32_nocopy_tn_64x40_4x8[SIMD16 {32; 1; 1} {64; 8; 1}] [SIMD16 {32; 1; 1} {64; 8; 1}] | 892 | 113.81 | 92.66 / 316.72 |
| 4 | gemm_f16f16f32_nocopy_tn_32x48_8x4[SIMD16 {32; 1; 1} {128; 4; 1}] [SIMD16 {32; 1; 1} {128; 4; 1}] | 1459 | 99.00 | 45.10 / 152.00 |
| 5 | gemm_f16f16f32_nocopy_tn_32x64_8x2x2[SIMD16 {5; 6; 1} {128; 2; 2}] [SIMD16 {5; 6; 1} {128; 2; 2}] | 1063 | 84.86 | 78.54 / 106.35 |
| 6 | gemm_f16f16f32_nocopy_tn_32x64_8x2x2[SIMD16 {5; 4; 1} {128; 2; 2}] [SIMD16 {5; 4; 1} {128; 2; 2}] | 1101 | 73.92 | 62.71 / 89.58 |
| 7 | gemm_f16f16f32_nocopy_tn_32x48_8x4[SIMD16 {5; 6; 1} {128; 4; 1}] [SIMD16 {5; 6; 1} {128; 4; 1}] | 1073 | 72.13 | 64.27 / 88.33 |
| 8 | ep_copy_uncached [SIMD32 {64; 1; 1} {256; 1; 1}] | 144 | 68.07 | 471.61 / 601.20 |
| 9 | gemm_f16f16f32_nocopy_tn_32x32_4x4x2[SIMD16 {10; 3; 1} {64; 4; 2}] [SIMD16 {10; 3; 1} {64; 4; 2}] | 1577 | 67.73 | 40.42 / 54.83 |
| 10 | nightg::k_night_dq32<21, 20>[SIMD32 {2400; 1; 1} {256; 1; 1}] [SIMD32 {2400; 1; 1} {256; 1; 1}] | 2398 | 59.35 | 23.75 / 30.31 |

### Recorded host Level Zero calls

| Rank | API | Calls | Summed duration ms | p50 / p95 us |
| ---: | --- | ---: | ---: | ---: |
| 1 | zeCommandListAppendMemoryCopy | 1541 | 21918.46 | 46.48 / 53223.94 |
| 2 | zeCommandListHostSynchronize | 388 | 1524.08 | 4.26 / 6556.45 |
| 3 | zeCommandListAppendLaunchKernelWithArguments | 96397 | 653.00 | 6.51 / 8.63 |
| 4 | zeMemGetAllocProperties | 218204 | 19.92 | 0.07 / 0.16 |
| 5 | zeKernelSetGroupSize | 96397 | 9.11 | 0.08 / 0.20 |
| 6 | zeEventHostSynchronize | 284 | 3.55 | 9.22 / 45.24 |
| 7 | zeModuleCreate | 5 | 1.25 | 207.20 / 326.59 |
| 8 | zeKernelSuggestGroupSize | 1883 | 1.08 | 0.33 / 2.33 |
| 9 | zeCommandListAppendBarrier | 146 | 0.53 | 3.47 / 4.75 |
| 10 | zeCommandListImmediateAppendCommandListsExp | 13 | 0.34 | 16.82 / 48.47 |

## B7

### Card 0

| Rank | Operation | Launches | Recorded service ms | p50 / p95 us |
| ---: | --- | ---: | ---: | ---: |
| 1 | gr_q8::launch_down<8> [SIMD16 {120; 1; 1} {256; 1; 1}] | 52283 | 1670.58 | 31.77 / 35.83 |
| 2 | wait_flag_ge [SIMD32 {1; 1; 1} {1; 1; 1}] | 539 | 1527.59 | 2.08 / 25583.26 |
| 3 | QsaVectorCodes[SIMD32 {33; 2; 1} {256; 1; 1}] [SIMD32 {33; 2; 1} {256; 1; 1}] | 51744 | 1234.84 | 24.89 / 27.08 |
| 4 | gdn_step_norm_multi [SIMD32 {48; 4; 1} {32; 4; 1}] | 77616 | 1198.38 | 15.73 / 16.67 |
| 5 | gr_q8::launch_up<8> [SIMD16 {160; 1; 1} {320; 1; 1}] | 52283 | 968.39 | 18.54 / 20.73 |
| 6 | gdn_step_norm_groups [SIMD32 {48; 4; 1} {128; 4; 1}] | 19404 | 936.96 | 48.12 / 50.83 |
| 7 | launch_kreg<KregQ5K, 8, 2, 128> [SIMD16 {1280; 1; 1} {192; 1; 1}] | 2695 | 705.87 | 260.42 / 278.16 |
| 8 | launch_kreg<KregQ5K, 8, 8, 256> [SIMD16 {31040; 1; 1} {80; 1; 1}] | 539 | 665.59 | 1223.96 / 1258.54 |
| 9 | launch_down_q2_word_expand [SIMD32 {80; 80; 1} {256; 1; 1}] | 16170 | 653.12 | 40.10 / 51.46 |
| 10 | launch_kreg<KregQ6K, 8, 2, 128> [SIMD16 {1280; 1; 1} {192; 1; 1}] | 15631 | 627.74 | 39.79 / 44.06 |

### Card 1

| Rank | Operation | Launches | Recorded service ms | p50 / p95 us |
| ---: | --- | ---: | ---: | ---: |
| 1 | ep_wait [SIMD32 {1; 1; 1} {1; 1; 1}] | 25872 | 18741.99 | 492.29 / 1254.42 |
| 2 | launch_down_q2_word_expand [SIMD32 {80; 80; 1} {256; 1; 1}] | 16170 | 721.18 | 44.48 / 58.33 |
| 3 | launch_gu_port<17, 8, true> [SIMD32 {40; 80; 1} {256; 1; 1}] | 5390 | 647.45 | 119.06 / 159.49 |
| 4 | launch_gu_row_reuse<21> [SIMD16 {20; 80; 1} {256; 1; 1}] | 7007 | 633.65 | 88.54 / 124.00 |
| 5 | launch_down_rows_k<20, 8, 2> [SIMD16 {40; 80; 1} {256; 1; 1}] | 9702 | 625.27 | 63.33 / 93.33 |
| 6 | launch_gu_port<16, 8, true> [SIMD32 {40; 80; 1} {256; 1; 1}] | 4851 | 621.40 | 127.29 / 174.11 |
| 7 | launch_gu_row_reuse<22> [SIMD16 {20; 80; 1} {256; 1; 1}] | 5390 | 457.88 | 84.53 / 110.42 |
| 8 | launch_gu_row_reuse<18> [SIMD16 {20; 80; 1} {256; 1; 1}] | 3234 | 317.23 | 99.48 / 124.06 |
| 9 | ep_send_psum [SIMD32 {80; 1; 1} {64; 1; 1}] | 25872 | 282.00 | 10.73 / 11.98 |
| 10 | native_expert_grouped [SIMD32 {200; 1; 1} {256; 1; 1}] | 25872 | 75.89 | 2.92 / 3.65 |

### Recorded host Level Zero calls

| Rank | API | Calls | Summed duration ms | p50 / p95 us |
| ---: | --- | ---: | ---: | ---: |
| 1 | zeCommandListHostSynchronize | 7168 | 28435.40 | 137.44 / 43790.48 |
| 2 | zeCommandListImmediateAppendCommandListsExp | 3773 | 107.27 | 13.84 / 63.85 |
| 3 | zeCommandListAppendMemoryCopy | 2229 | 34.82 | 9.62 / 35.84 |
| 4 | zeModuleCreate | 27 | 23.80 | 412.56 / 2879.15 |
| 5 | zeEventHostSynchronize | 5926 | 21.62 | 1.21 / 4.78 |
| 6 | zeCommandListAppendLaunchKernel | 6485 | 19.05 | 1.61 / 2.25 |
| 7 | zeCommandListCreate | 15 | 9.78 | 662.98 / 855.80 |
| 8 | zeKernelSetArgumentValue | 44318 | 4.17 | 0.07 / 0.24 |
| 9 | zeKernelSetGroupSize | 6497 | 0.31 | 0.04 / 0.07 |
| 10 | zeContextMakeMemoryResident | 2 | 0.31 | 155.29 / 176.23 |

