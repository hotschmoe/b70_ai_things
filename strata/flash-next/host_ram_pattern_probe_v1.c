/* Bounded anonymous CPU memory diagnostic; no GPU or model source access. */
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static double now(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return value.tv_sec + value.tv_nsec * 1e-9;
}

static void residency(void) {
    FILE *file = fopen("/proc/self/status", "r");
    if (!file) return;
    char line[256];
    while (fgets(line, sizeof(line), file))
        if (!strncmp(line, "VmRSS:", 6) || !strncmp(line, "VmSwap:", 7))
            fputs(line, stdout);
    fclose(file);
}

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    char *end = NULL;
    errno = 0;
    uint64_t bytes = strtoull(argv[1], &end, 10);
    if (errno || !end || *end || !bytes || bytes % 4096 || bytes > (UINT64_C(64) << 30)) return 2;
    void *memory = NULL;
    if (posix_memalign(&memory, 4096, (size_t)bytes)) return 3;
    const unsigned char patterns[] = {0x00, 0xff, 0x55, 0xaa, 0x01, 0xfe, 0xdf, 0x20};
    const char *injection = getenv("B70_RAM_PROBE_INJECT_OFFSET");
    uint64_t inject_offset = 0;
    if (injection) {
        inject_offset = strtoull(injection, &end, 10);
        if (!end || *end || inject_offset >= bytes) { free(memory); return 2; }
    }
    printf("CONFIG bytes=%" PRIu64 " anonymous=1 gpu=0 source_access=0 injected=%d\n", bytes, injection != NULL);
    fflush(stdout);
    for (size_t pass = 0; pass < sizeof(patterns); ++pass) {
        double started = now();
        unsigned char pattern = patterns[pass];
        uint64_t expected = UINT64_C(0x0101010101010101) * pattern;
        memset(memory, pattern, (size_t)bytes);
        if (injection && pass == 0) ((unsigned char *)memory)[inject_offset] ^= 0x20;
        residency();
        volatile const uint64_t *words = memory;
        uint64_t errors = 0;
        for (uint64_t word = 0; word < bytes / 8; ++word) {
            uint64_t actual = words[word];
            if (actual != expected) {
                if (errors < 64) {
                    uint64_t offset = word * 8;
                    printf("MISMATCH pass=%zu offset=%" PRIu64 " page_offset=%" PRIu64 " expected=%016" PRIx64 " actual=%016" PRIx64 " xor=%016" PRIx64 "\n",
                           pass, offset, offset % 4096, expected, actual, expected ^ actual);
                }
                ++errors;
            }
        }
        printf("RESULT pass=%zu pattern=%02x bytes=%" PRIu64 " errors=%" PRIu64 " elapsed=%.6f\n", pass, pattern, bytes, errors, now() - started);
        fflush(stdout);
        if (errors) {
            puts("VERDICT observed_mismatch=1 physical_PFN_unobserved=1 no_hardware_cause_assignment=1");
            free(memory);
            return 1;
        }
    }
    free(memory);
    puts("VERDICT bounded_patterns_pass=1 full_host_stability_unproved=1 GPU_driver_paths_unobserved=1 physical_PFN_unobserved=1");
    return 0;
}
