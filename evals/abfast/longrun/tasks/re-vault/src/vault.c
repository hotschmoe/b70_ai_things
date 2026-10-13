/* Reference program for task re-vault. NOT visible to the agent (only the stripped binary is). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <sys/stat.h>
#include <unistd.h>

#define HDR 28
static uint32_t crc_tab[256];
static void crc_init(void) {
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t c = i;
        for (int k = 0; k < 8; k++) c = (c & 1) ? (c >> 1) ^ 0x82F63B78u : c >> 1;
        crc_tab[i] = c;
    }
}
static uint32_t vcrc(const uint8_t *p, size_t n) {
    uint32_t c = 0xA5A5A5A5u;
    for (size_t i = 0; i < n; i++) c = crc_tab[(c ^ p[i]) & 0xff] ^ (c >> 8);
    return c ^ 0x3C3C3C3Cu;
}
static uint64_t fnv64(const uint8_t *p, size_t n, uint64_t h) {
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 0x100000001b3ULL; }
    return h;
}
#define FNV_OFF 0xcbf29ce484222325ULL
static uint32_t icheck(const uint8_t *p, size_t n) {
    uint32_t h = 0x811C9DC5u;
    for (size_t i = 0; i < n; i++) { h = ((h << 5) | (h >> 27)) ^ p[i]; h *= 0x01000193u; h ^= h >> 15; }
    return h;
}
static void ks_init(uint64_t *st, const char *key, uint64_t salt, uint32_t idx) {
    uint64_t s = fnv64((const uint8_t *)key, strlen(key), FNV_OFF) ^ salt ^ ((uint64_t)idx * 0x9E3779B97F4A7C15ULL);
    if (!s) s = 0x1234567ULL;
    *st = s;
}
static uint8_t ks_next(uint64_t *st) {
    uint64_t x = *st;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    *st = x;
    return (uint8_t)((x * 0x2545F4914F6CDD1DULL) >> 56);
}
static uint32_t keycheck(const char *key, uint64_t salt) {
    uint64_t h = fnv64((const uint8_t *)key, strlen(key), FNV_OFF ^ salt);
    return (uint32_t)(h ^ (h >> 32)) ^ 0xB16B00B5u;
}

/* ---- codecs ---- */
static size_t rle_enc(const uint8_t *in, size_t n, uint8_t *out) {
    size_t i = 0, o = 0, lit = 0, lit_start = 0;
    while (i < n) {
        size_t r = 1;
        while (i + r < n && in[i + r] == in[i] && r < 130) r++;
        if (r >= 3) {
            if (lit) { out[o++] = (uint8_t)(lit - 1); memcpy(out + o, in + lit_start, lit); o += lit; lit = 0; }
            out[o++] = (uint8_t)(0x80 | (r - 3)); out[o++] = in[i]; i += r;
        } else {
            if (!lit) lit_start = i;
            lit++; i++;
            if (lit == 128) { out[o++] = 127; memcpy(out + o, in + lit_start, lit); o += lit; lit = 0; }
        }
    }
    if (lit) { out[o++] = (uint8_t)(lit - 1); memcpy(out + o, in + lit_start, lit); o += lit; }
    return o;
}
static int rle_dec(const uint8_t *in, size_t n, uint8_t *out, size_t rawn) {
    size_t i = 0, o = 0;
    while (i < n) {
        uint8_t c = in[i++];
        if (c < 0x80) { size_t l = (size_t)c + 1; if (i + l > n || o + l > rawn) return -1; memcpy(out + o, in + i, l); i += l; o += l; }
        else { size_t l = (size_t)(c & 0x7f) + 3; if (i >= n || o + l > rawn) return -1; memset(out + o, in[i++], l); o += l; }
    }
    return o == rawn ? 0 : -1;
}
static size_t lz_enc(const uint8_t *in, size_t n, uint8_t *out) {
    size_t i = 0, o = 0, flag_at = 0; int bit = 8;
    while (i < n) {
        if (bit == 8) { flag_at = o++; out[flag_at] = 0; bit = 0; }
        size_t best = 0, bd = 0;
        size_t lo = i > 4095 ? i - 4095 : 0;
        for (size_t j = i; j-- > lo;) {
            size_t l = 0;
            while (l < 18 && i + l < n && in[j + l] == in[i + l]) l++;
            if (l > best) { best = l; bd = i - j; }
            if (best == 18) break;
        }
        if (best == 3 && bd > 2048) best = 0;
        if (best >= 3) {
            out[o++] = (uint8_t)(bd & 0xff);
            out[o++] = (uint8_t)(((bd >> 8) << 4) | (best - 3));
            i += best;
        } else {
            out[flag_at] |= (uint8_t)(1 << bit);
            out[o++] = in[i++];
        }
        bit++;
    }
    return o;
}
static int lz_dec(const uint8_t *in, size_t n, uint8_t *out, size_t rawn) {
    size_t i = 0, o = 0;
    while (o < rawn) {
        if (i >= n) return -1;
        uint8_t fl = in[i++];
        for (int b = 0; b < 8 && o < rawn; b++) {
            if (fl & (1 << b)) { if (i >= n) return -1; out[o++] = in[i++]; }
            else {
                if (i + 2 > n) return -1;
                size_t d = in[i] | ((size_t)(in[i + 1] >> 4) << 8), l = (in[i + 1] & 15) + 3; i += 2;
                if (d == 0 || d > o || o + l > rawn) return -1;
                for (size_t k = 0; k < l; k++) { out[o] = out[o - d]; o++; }
            }
        }
    }
    return 0;
}

/* ---- archive model ---- */
typedef struct { char *name; uint8_t method; uint16_t mode; uint32_t raw, comp, off, crc; } Ent;
typedef struct { uint8_t *buf; size_t len; uint16_t flags; uint32_t count, ioff; uint64_t salt; uint32_t kc; Ent *e; int idx_ok; } Arc;

static uint16_t rd16(const uint8_t *p) { return p[0] | (p[1] << 8); }
static uint32_t rd32(const uint8_t *p) { return p[0] | (p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24); }
static uint64_t rd64(const uint8_t *p) { return rd32(p) | ((uint64_t)rd32(p + 4) << 32); }
static void wr16(uint8_t *p, uint16_t v) { p[0] = v; p[1] = v >> 8; }
static void wr32(uint8_t *p, uint32_t v) { for (int i = 0; i < 4; i++) p[i] = v >> (8 * i); }
static void wr64(uint8_t *p, uint64_t v) { for (int i = 0; i < 8; i++) p[i] = v >> (8 * i); }

static int die(int code, const char *fmt, const char *a) { fprintf(stderr, "vault: "); fprintf(stderr, fmt, a); fprintf(stderr, "\n"); return code; }

static uint8_t *slurp(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb"); if (!f) return NULL;
    size_t cap = 4096, n = 0; uint8_t *b = malloc(cap);
    for (;;) { size_t r = fread(b + n, 1, cap - n, f); n += r; if (n < cap) break; cap *= 2; b = realloc(b, cap); }
    fclose(f); *len = n; return b;
}

static int arc_open(Arc *a, const char *path) {
    memset(a, 0, sizeof *a);
    a->buf = slurp(path, &a->len);
    if (!a->buf) return die(6, "cannot read %s", path);
    if (a->len < HDR || memcmp(a->buf, "VLT3", 4)) return die(3, "%s: not a vault archive", path);
    if (rd16(a->buf + 4) != 3) return die(3, "%s: unsupported version", path);
    a->flags = rd16(a->buf + 6); a->count = rd32(a->buf + 8); a->ioff = rd32(a->buf + 12);
    a->salt = rd64(a->buf + 16); a->kc = rd32(a->buf + 24);
    if (a->ioff < HDR || a->ioff > a->len) return die(3, "%s: corrupt index offset", path);
    const uint8_t *p = a->buf + a->ioff, *end = a->buf + a->len;
    a->e = calloc(a->count + 1, sizeof(Ent));
    for (uint32_t i = 0; i < a->count; i++) {
        if (p + 2 > end) return die(3, "%s: truncated index", path);
        uint16_t nl = rd16(p); p += 2;
        if (p + nl + 19 > end) return die(3, "%s: truncated index", path);
        Ent *e = &a->e[i]; e->name = malloc(nl + 1); memcpy(e->name, p, nl); e->name[nl] = 0; p += nl;
        e->method = *p++; e->mode = rd16(p); p += 2; e->raw = rd32(p); e->comp = rd32(p + 4); e->off = rd32(p + 8); e->crc = rd32(p + 12); p += 16;
        if ((uint64_t)e->off + e->comp > a->ioff || e->off < HDR) return die(3, "%s: entry out of range", path);
    }
    if (p + 4 > end) return die(3, "%s: truncated index", path);
    a->idx_ok = icheck(a->buf + a->ioff, p - (a->buf + a->ioff)) == rd32(p);
    return 0;
}

/* returns 0 ok; fills *out (malloc) / *outn. status: 1 bad key, 2 bad data */
static int ent_data(Arc *a, uint32_t i, const char *key, uint8_t **out, size_t *outn) {
    Ent *e = &a->e[i];
    uint8_t *c = malloc(e->comp + 1); memcpy(c, a->buf + e->off, e->comp);
    if (a->flags & 1) {
        if (!key) { free(c); return 1; }
        uint64_t st; ks_init(&st, key, a->salt, i);
        for (uint32_t k = 0; k < e->comp; k++) c[k] ^= ks_next(&st);
    }
    uint8_t *r = malloc(e->raw + 1); int rc = 0;
    if (e->method == 0) { if (e->comp != e->raw) rc = 2; else memcpy(r, c, e->raw); }
    else if (e->method == 1) rc = rle_dec(c, e->comp, r, e->raw) ? 2 : 0;
    else if (e->method == 2) rc = lz_dec(c, e->comp, r, e->raw) ? 2 : 0;
    else rc = 2;
    free(c);
    if (rc) { free(r); return rc; }
    *out = r; *outn = e->raw; return 0;
}
static int check_key(Arc *a, const char *key) {
    if (!(a->flags & 1)) return 0;
    if (!key) return die(4, "archive is encrypted; key required%s", "");
    if (keycheck(key, a->salt) != a->kc) return die(4, "wrong key%s", "");
    return 0;
}

static int mkdirs(char *path) {
    for (char *p = path + 1; *p; p++) if (*p == '/') { *p = 0; mkdir(path, 0755); *p = '/'; }
    return 0;
}
static int unsafe(const char *n) {
    if (!*n || n[0] == '/') return 1;
    const char *p = n;
    while (*p) {
        const char *q = strchr(p, '/'); size_t l = q ? (size_t)(q - p) : strlen(p);
        if (l == 2 && p[0] == '.' && p[1] == '.') return 1;
        if (l == 0) return 1;
        if (!q) break; p = q + 1;
    }
    return 0;
}

/* ---- commands ---- */
static const char *MN[] = {"store", "rle", "lz"};

static int cmd_digest(int argc, char **argv) {
    if (argc != 1) return die(2, "usage: vault digest FILE%s", "");
    size_t n; uint8_t *b = slurp(argv[0], &n);
    if (!b) return die(6, "cannot read %s", argv[0]);
    printf("crc=%08x fnv64=%016llx len=%zu\n", vcrc(b, n), (unsigned long long)fnv64(b, n, FNV_OFF), n);
    return 0;
}
static int cmd_info(int argc, char **argv) {
    if (argc != 1) return die(2, "usage: vault info ARCHIVE%s", "");
    Arc a; int rc = arc_open(&a, argv[0]); if (rc) return rc;
    uint64_t raw = 0, comp = 0;
    for (uint32_t i = 0; i < a.count; i++) { raw += a.e[i].raw; comp += a.e[i].comp; }
    printf("format: VLT3 v3\nentries: %u\nencrypted: %s\nindex offset: %u\nindex check: %s\nraw bytes: %llu\npacked bytes: %llu\n",
           a.count, (a.flags & 1) ? "yes" : "no", a.ioff, a.idx_ok ? "OK" : "BAD", (unsigned long long)raw, (unsigned long long)comp);
    return 0;
}
static int cmd_list(int argc, char **argv) {
    int lng = 0;
    if (argc && !strcmp(argv[0], "-l")) { lng = 1; argc--; argv++; }
    if (argc != 1) return die(2, "usage: vault list [-l] ARCHIVE%s", "");
    Arc a; int rc = arc_open(&a, argv[0]); if (rc) return rc;
    uint64_t raw = 0, comp = 0;
    for (uint32_t i = 0; i < a.count; i++) {
        Ent *e = &a.e[i]; raw += e->raw; comp += e->comp;
        if (lng) printf("%04o %-5s %8u %8u %08x %s\n", e->mode & 0777, e->method < 3 ? MN[e->method] : "?", e->raw, e->comp, e->crc, e->name);
        else printf("%s\n", e->name);
    }
    if (lng) printf("%u entries, %llu bytes (%llu packed)\n", a.count, (unsigned long long)raw, (unsigned long long)comp);
    return 0;
}
static int cmd_cat(int argc, char **argv) {
    const char *key = NULL;
    if (argc >= 2 && !strcmp(argv[0], "-k")) { key = argv[1]; argc -= 2; argv += 2; }
    if (argc != 2) return die(2, "usage: vault cat [-k KEY] ARCHIVE NAME%s", "");
    Arc a; int rc = arc_open(&a, argv[0]); if (rc) return rc;
    if ((rc = check_key(&a, key))) return rc;
    for (uint32_t i = 0; i < a.count; i++) if (!strcmp(a.e[i].name, argv[1])) {
        uint8_t *d; size_t n; int s = ent_data(&a, i, key, &d, &n);
        if (s) return die(3, "%s: corrupt entry", argv[1]);
        if (vcrc(d, n) != a.e[i].crc) return die(3, "%s: checksum mismatch", argv[1]);
        fwrite(d, 1, n, stdout); return 0;
    }
    return die(5, "%s: no such entry", argv[1]);
}
static int cmd_extract(int argc, char **argv) {
    const char *key = NULL;
    if (argc >= 2 && !strcmp(argv[0], "-k")) { key = argv[1]; argc -= 2; argv += 2; }
    if (argc < 2) return die(2, "usage: vault extract [-k KEY] ARCHIVE DIR [NAME...]%s", "");
    Arc a; int rc = arc_open(&a, argv[0]); if (rc) return rc;
    if ((rc = check_key(&a, key))) return rc;
    for (int k = 2; k < argc; k++) {
        int f = 0; for (uint32_t i = 0; i < a.count; i++) if (!strcmp(a.e[i].name, argv[k])) f = 1;
        if (!f) return die(5, "%s: no such entry", argv[k]);
    }
    for (uint32_t i = 0; i < a.count; i++) {
        Ent *e = &a.e[i];
        if (argc > 2) { int want = 0; for (int k = 2; k < argc; k++) if (!strcmp(e->name, argv[k])) want = 1; if (!want) continue; }
        if (unsafe(e->name)) return die(3, "%s: unsafe entry name", e->name);
        uint8_t *d; size_t n; int s = ent_data(&a, i, key, &d, &n);
        if (s) return die(3, "%s: corrupt entry", e->name);
        if (vcrc(d, n) != e->crc) return die(3, "%s: checksum mismatch", e->name);
        char path[4096]; snprintf(path, sizeof path, "%s/%s", argv[1], e->name);
        mkdirs(path);
        FILE *f = fopen(path, "wb"); if (!f) return die(6, "cannot write %s", path);
        fwrite(d, 1, n, f); fclose(f); chmod(path, e->mode & 0777);
        printf("x %s\n", e->name);
    }
    return 0;
}
static int cmd_verify(int argc, char **argv) {
    const char *key = NULL;
    if (argc >= 2 && !strcmp(argv[0], "-k")) { key = argv[1]; argc -= 2; argv += 2; }
    if (argc != 1) return die(2, "usage: vault verify [-k KEY] ARCHIVE%s", "");
    Arc a; int rc = arc_open(&a, argv[0]); if (rc) return rc;
    if ((rc = check_key(&a, key))) return rc;
    int bad = 0;
    if (!a.idx_ok) { printf("BAD index (checksum mismatch)\n"); bad++; }
    for (uint32_t i = 0; i < a.count; i++) {
        uint8_t *d; size_t n; int s = ent_data(&a, i, key, &d, &n);
        if (s) { printf("BAD %s (corrupt data)\n", a.e[i].name); bad++; }
        else if (vcrc(d, n) != a.e[i].crc) { printf("BAD %s (crc mismatch)\n", a.e[i].name); bad++; }
        else printf("OK %s\n", a.e[i].name);
    }
    if (bad) { printf("FAILED %d problem%s\n", bad, bad == 1 ? "" : "s"); return 1; }
    printf("verified %u entries\n", a.count);
    return 0;
}
static int cmd_pack(int argc, char **argv) {
    const char *key = NULL; int method = -1;
    while (argc >= 2 && argv[0][0] == '-') {
        if (!strcmp(argv[0], "-k")) key = argv[1];
        else if (!strcmp(argv[0], "-m")) {
            if (!strcmp(argv[1], "auto")) method = -1;
            else if (!strcmp(argv[1], "0") || !strcmp(argv[1], "1") || !strcmp(argv[1], "2")) method = argv[1][0] - '0';
            else return die(2, "bad method %s", argv[1]);
        } else return die(2, "unknown option %s", argv[0]);
        argc -= 2; argv += 2;
    }
    if (argc < 2) return die(2, "usage: vault pack [-k KEY] [-m 0|1|2|auto] OUT FILE...%s", "");
    const char *outp = argv[0]; int nf = argc - 1; char **fn = argv + 1;
    char **names = calloc(nf, sizeof *names);
    for (int i = 0; i < nf; i++) {
        const char *n = fn[i]; while (!strncmp(n, "./", 2)) n += 2;
        if (unsafe(n)) return die(3, "%s: unsafe path", fn[i]);
        for (int j = 0; j < i; j++) if (!strcmp(names[j], n)) return die(2, "%s: duplicate entry", n);
        names[i] = (char *)n;
    }
    uint64_t salt = 0;
    if (key) { salt = FNV_OFF; for (int i = 0; i < nf; i++) salt = fnv64((const uint8_t *)names[i], strlen(names[i]), salt); salt ^= nf; }
    size_t cap = HDR; uint8_t *out = malloc(1 << 20); size_t outcap = 1 << 20; (void)cap;
    size_t o = HDR;
    Ent *es = calloc(nf, sizeof *es);
    for (int i = 0; i < nf; i++) {
        size_t n; uint8_t *d = slurp(fn[i], &n);
        if (!d) return die(6, "cannot read %s", fn[i]);
        struct stat st; stat(fn[i], &st);
        uint8_t *best = malloc(n * 2 + 64), *tmp = malloc(n * 2 + 64); size_t bn = n; int bm = 0;
        memcpy(best, d, n);
        for (int m = 1; m <= 2; m++) {
            if (method >= 0 && m != method) continue;
            size_t tn = m == 1 ? rle_enc(d, n, tmp) : lz_enc(d, n, tmp);
            if (method >= 0 || tn < bn) { memcpy(best, tmp, tn); bn = tn; bm = m; }
        }
        if (method == 0) { bm = 0; bn = n; memcpy(best, d, n); }
        es[i] = (Ent){names[i], (uint8_t)bm, (uint16_t)(st.st_mode & 0777), (uint32_t)n, (uint32_t)bn, (uint32_t)o, vcrc(d, n)};
        if (key) { uint64_t s; ks_init(&s, key, salt, i); for (size_t k = 0; k < bn; k++) best[k] ^= ks_next(&s); }
        while (o + bn + 65536 > outcap) { outcap *= 2; out = realloc(out, outcap); }
        memcpy(out + o, best, bn); o += bn;
        free(d); free(best); free(tmp);
    }
    size_t ioff = o;
    for (int i = 0; i < nf; i++) {
        size_t nl = strlen(es[i].name);
        while (o + nl + 64 > outcap) { outcap *= 2; out = realloc(out, outcap); }
        wr16(out + o, nl); o += 2; memcpy(out + o, es[i].name, nl); o += nl;
        out[o++] = es[i].method; wr16(out + o, es[i].mode); o += 2;
        wr32(out + o, es[i].raw); wr32(out + o + 4, es[i].comp); wr32(out + o + 8, es[i].off); wr32(out + o + 12, es[i].crc); o += 16;
    }
    wr32(out + o, icheck(out + ioff, o - ioff)); o += 4;
    memcpy(out, "VLT3", 4); wr16(out + 4, 3); wr16(out + 6, key ? 1 : 0); wr32(out + 8, nf); wr32(out + 12, ioff);
    wr64(out + 16, salt); wr32(out + 24, key ? keycheck(key, salt) : 0);
    FILE *f = fopen(outp, "wb"); if (!f) return die(6, "cannot write %s", outp);
    fwrite(out, 1, o, f); fclose(f);
    printf("packed %d entr%s, %zu bytes\n", nf, nf == 1 ? "y" : "ies", o);
    return 0;
}

int main(int argc, char **argv) {
    crc_init();
    if (argc < 2) return die(2, "usage: vault pack|list|info|cat|extract|verify|digest ...%s", "");
    const char *c = argv[1]; argc -= 2; argv += 2;
    if (!strcmp(c, "pack")) return cmd_pack(argc, argv);
    if (!strcmp(c, "list")) return cmd_list(argc, argv);
    if (!strcmp(c, "info")) return cmd_info(argc, argv);
    if (!strcmp(c, "cat")) return cmd_cat(argc, argv);
    if (!strcmp(c, "extract")) return cmd_extract(argc, argv);
    if (!strcmp(c, "verify")) return cmd_verify(argc, argv);
    if (!strcmp(c, "digest")) return cmd_digest(argc, argv);
    return die(2, "unknown command %s", c);
}
