/* Reference program for task re-vm. NOT visible to the agent (only the stripped binary is). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>

typedef struct { int t; int64_t v; } V;                    /* t: 0 int, 1 str, 2 arr (v = heap handle) */
typedef struct { int len; char *s; int64_t *a; } Obj;
static Obj heap[2048]; static int nheap; static long cells;
static V stk[256]; static int sp;
static V glob[64];
typedef struct { V loc[16]; int base; uint32_t ret; } Fr;
static Fr fr[33]; static int nfr;
static uint8_t *code; static uint32_t codelen;
static int ctype[65536]; static int64_t cint[65536]; static char *cstr[65536]; static int clen[65536]; static int nconst;
static uint8_t perm[256], inv[256];
static int obf;
static uint32_t cur_pc;

static void trap(int rc, const char *msg) {
    fflush(stdout);
    fprintf(stderr, "vm: %s at pc=%u\n", msg, cur_pc);
    exit(rc);
}
static void die(int rc, const char *fmt, const char *a, long b) {
    fprintf(stderr, "vmrun: "); fprintf(stderr, fmt, a, b); fprintf(stderr, "\n"); exit(rc);
}
static void init_perm(void) {
    uint32_t st = 0xB70; for (int i = 0; i < 256; i++) perm[i] = i;
    for (int i = 255; i > 0; i--) {
        st = st * 1103515245u + 12345u; int j = (st >> 16) % (i + 1);
        uint8_t t = perm[i]; perm[i] = perm[j]; perm[j] = t;
    }
    for (int i = 0; i < 256; i++) inv[perm[i]] = i;
}
static uint32_t csum(const uint8_t *p, size_t n) {
    uint32_t h = 0x1505;
    for (size_t i = 0; i < n; i++) h = (h * 33) ^ p[i];
    return ((h << 7) | (h >> 25)) ^ 0xC0DEC0DEu;
}
enum { NOP, PUSHI, PUSHC, POP, DUP, SWAP, OVER, ROT, ADD, SUB, MUL, DIV, MOD, NEG, AND, OR, XOR, NOT, SHL, SHR, USHR,
       EQ, NE, LT, LE, GT, GE, JMP, JZ, JNZ, CALL, RET, HALT, LOAD, STORE, GLOAD, GSTORE, NEWARR, ALOAD, ASTORE, ALEN,
       SLEN, SCAT, SIDX, SSUB, ITOS, STOI, CHR, PRINT, PUTC, READ, READC, ASSERT, DEPTH, PUSHL, TYPEOF, NOPS };
static const uint8_t oplen[NOPS] = {0,4,2,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0, 2,2,2, 3,0,0, 1,1,1,1, 0,0,0,0,
    0,0,0,0,0,0,0, 0,0,0,0,0,0, 8,0};

static uint8_t dec(uint32_t pc, int k) {   /* k = -1: opcode byte, else operand index */
    uint8_t b = code[pc + 1 + k];
    if (!obf) return b;
    return k < 0 ? (uint8_t)(b ^ ((pc * 7 + 3) & 0xff)) : (uint8_t)(b ^ ((pc + 13 * k + 0x5A) & 0xff));
}
static uint8_t rawop(uint32_t pc) { uint8_t b = code[pc]; return obf ? (uint8_t)(b ^ ((pc * 7 + 3) & 0xff)) : b; }

static V pop(void) {
    if (sp <= fr[nfr].base) trap(64, "stack underflow");
    return stk[--sp];
}
static void push(V v) { if (sp >= 256) trap(65, "stack overflow"); stk[sp++] = v; }
static V I(int64_t x) { V v = {0, x}; return v; }
static int64_t popi(void) { V v = pop(); if (v.t) trap(66, "type error"); return v.v; }
static V newobj(int t, int len) {
    if (nheap >= 2048) trap(72, "heap exhausted");
    Obj *o = &heap[nheap];
    if (t == 2) { if (cells + len > 65536) trap(72, "heap exhausted"); cells += len; o->a = calloc(len + 1, 8); }
    else o->s = calloc(len + 1, 1);
    o->len = len; V v = {t, nheap++}; return v;
}
static int cmpv(V a, V b) {
    if (a.t != b.t) trap(66, "type error");
    if (a.t == 0) return a.v < b.v ? -1 : a.v > b.v;
    if (a.t == 1) {
        Obj *x = &heap[a.v], *y = &heap[b.v]; int n = x->len < y->len ? x->len : y->len;
        int c = memcmp(x->s, y->s, n); if (c) return c < 0 ? -1 : 1;
        return x->len < y->len ? -1 : x->len > y->len;
    }
    return 2;   /* arrays: only identity comparable */
}



int main(int argc, char **argv) {
    init_perm();
    long budget = 1000000; int info = 0; const char *path = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-b")) {
            if (i + 1 >= argc) die(1, "bad budget%s", "", 0);
            char *e; long b = strtol(argv[++i], &e, 10);
            if (*e || b <= 0) die(1, "bad budget%s", "", 0);
            budget = b;
        } else if (!strcmp(argv[i], "--info")) info = 1;
        else if (argv[i][0] == '-' && argv[i][1]) die(1, "unknown option %s", argv[i], 0);
        else if (!path) path = argv[i];
        else path = (const char *)-1;
    }
    if (!path || path == (const char *)-1) { fprintf(stderr, "usage: vmrun [-b BUDGET] [--info] PROGRAM.bc\n"); return 1; }
    FILE *f = fopen(path, "rb"); if (!f) die(2, "cannot open %s", path, 0);
    size_t cap = 1 << 16, n = 0; uint8_t *buf = malloc(cap);
    for (;;) { size_t r = fread(buf + n, 1, cap - n, f); n += r; if (n < cap) break; cap *= 2; buf = realloc(buf, cap); }
    fclose(f);
    if (n < 4 || memcmp(buf, "BCX1", 4)) die(3, "bad magic%s", "", 0);
    if (n < 16) die(3, "truncated file%s", "", 0);
    if (buf[4] != 2) die(3, "unsupported version %.0s%ld", "", (long)buf[4]);
    if (n < 20) die(3, "truncated file%s", "", 0);
    uint32_t want = buf[n - 4] | (buf[n - 3] << 8) | (buf[n - 2] << 16) | ((uint32_t)buf[n - 1] << 24);
    if (csum(buf, n - 4) != want) die(4, "checksum mismatch%s", "", 0);
    int flags = buf[5]; obf = flags & 1;
    nconst = buf[6] | (buf[7] << 8);
    codelen = buf[8] | (buf[9] << 8) | (buf[10] << 16) | ((uint32_t)buf[11] << 24);
    uint32_t entry = buf[12] | (buf[13] << 8) | (buf[14] << 16) | ((uint32_t)buf[15] << 24);
    size_t p = 16, end = n - 4; int nint = 0, nstr = 0;
    for (int i = 0; i < nconst; i++) {
        if (p >= end) die(3, "malformed file%s", "", 0);
        int tag = buf[p++];
        if (tag == 0) {
            if (p + 8 > end) die(3, "malformed file%s", "", 0);
            uint64_t v = 0; for (int k = 7; k >= 0; k--) v = (v << 8) | buf[p + k];
            p += 8; ctype[i] = 0; cint[i] = (int64_t)v; nint++;
        } else if (tag == 1) {
            if (p + 2 > end) die(3, "malformed file%s", "", 0);
            int l = buf[p] | (buf[p + 1] << 8); p += 2;
            if (l > 4096 || p + l > end) die(3, "malformed file%s", "", 0);
            char *s = malloc(l + 1);
            for (int j = 0; j < l; j++) s[j] = (flags & 2) ? (char)(buf[p + j] ^ ((0x21 + i * 3 + j) & 0xff)) : (char)buf[p + j];
            s[l] = 0; p += l; ctype[i] = 1; cstr[i] = s; clen[i] = l; nstr++;
        } else die(3, "bad constant tag%s", "", 0);
    }
    if (end - p != codelen) die(3, "malformed file%s", "", 0);
    if (entry > codelen) die(3, "bad entry point%s", "", 0);
    code = buf + p;
    if (info) {
        printf("version: 2\nflags: 0x%02x\nconstants: %d (%d int, %d str)\ncode: %u bytes\nentry: %u\nchecksum: %08x\n",
               flags, nconst, nint, nstr, codelen, entry, want);
        return 0;
    }
    uint32_t pc = entry; long used = 0;
    for (;;) {
        cur_pc = pc;
        if (pc == codelen) { fflush(stdout); return 0; }
        if (used >= budget) trap(70, "instruction budget exceeded");
        used++;
        uint8_t rb = rawop(pc), op = obf ? inv[rb] : rb;
        if (op >= NOPS) trap(69, "illegal instruction");
        if (pc + 1 + oplen[op] > codelen) trap(69, "truncated instruction");
        int64_t imm = 0; uint32_t npc = pc + 1 + oplen[op];
        if (op == PUSHI) { uint32_t u = 0; for (int k = 3; k >= 0; k--) u = (u << 8) | dec(pc, k); imm = (int32_t)u; }
        else if (op == PUSHL) { uint64_t u = 0; for (int k = 7; k >= 0; k--) u = (u << 8) | dec(pc, k); imm = (int64_t)u; }
        else if (op == PUSHC || op == JMP || op == JZ || op == JNZ || op == CALL) {
            uint32_t u = dec(pc, 0) | (dec(pc, 1) << 8); imm = (op == JMP || op == JZ || op == JNZ) ? (int64_t)(int16_t)u : (int64_t)u;
        } else if (oplen[op] == 1) imm = dec(pc, 0);
        V a, b, c;
        switch (op) {
        case NOP: break;
        case PUSHI: case PUSHL: push(I(imm)); break;
        case PUSHC:
            if (imm >= nconst) trap(76, "bad constant index");
            if (ctype[imm] == 0) push(I(cint[imm]));
            else { V s = newobj(1, clen[imm]); memcpy(heap[s.v].s, cstr[imm], clen[imm]); push(s); }
            break;
        case POP: pop(); break;
        case DUP: a = pop(); push(a); push(a); break;
        case SWAP: b = pop(); a = pop(); push(b); push(a); break;
        case OVER: b = pop(); a = pop(); push(a); push(b); push(a); break;
        case ROT: c = pop(); b = pop(); a = pop(); push(b); push(c); push(a); break;
        case ADD: case SUB: case MUL: case DIV: case MOD: case AND: case OR: case XOR: case SHL: case SHR: case USHR: {
            if (sp - fr[nfr].base < 2) trap(64, "stack underflow");
            b = pop(); a = pop(); if (a.t || b.t) trap(66, "type error");
            uint64_t x = a.v, y = b.v; int64_t r = 0;
            switch (op) {
            case ADD: r = x + y; break; case SUB: r = x - y; break; case MUL: r = x * y; break;
            case DIV: if (!y) trap(67, "division by zero"); r = (a.v == INT64_MIN && b.v == -1) ? a.v : a.v / b.v; break;
            case MOD: if (!y) trap(67, "division by zero"); r = (b.v == -1) ? 0 : a.v % b.v; break;
            case AND: r = x & y; break; case OR: r = x | y; break; case XOR: r = x ^ y; break;
            case SHL: r = x << (y & 63); break; case SHR: r = a.v >> (y & 63); break; case USHR: r = x >> (y & 63); break;
            }
            push(I(r)); break; }
        case NEG: a = pop(); if (a.t) trap(66, "type error"); push(I((int64_t)(0 - (uint64_t)a.v))); break;
        case NOT: a = pop(); if (a.t) trap(66, "type error"); push(I(~a.v)); break;
        case EQ: case NE: case LT: case LE: case GT: case GE: {
            if (sp - fr[nfr].base < 2) trap(64, "stack underflow");
            b = pop(); a = pop(); int r;
            if (a.t == 2 && b.t == 2) { if (op != EQ && op != NE) trap(66, "type error"); r = (a.v == b.v) ? 0 : 2; }
            else r = cmpv(a, b);
            int res = 0;
            switch (op) { case EQ: res = r == 0; break; case NE: res = r != 0; break; case LT: res = r < 0; break;
                          case LE: res = r <= 0; break; case GT: res = r > 0; break; case GE: res = r >= 0; break; }
            push(I(res)); break; }
        case JMP: case JZ: case JNZ: {
            int take = 1;
            if (op != JMP) { int64_t v = popi(); take = (op == JZ) ? v == 0 : v != 0; }
            if (take) { int64_t t = (int64_t)npc + imm; if (t < 0 || t > (int64_t)codelen) trap(75, "jump out of range"); npc = (uint32_t)t; }
            break; }
        case CALL: {
            int nargs = dec(pc, 2);
            if (nargs > 16) trap(73, "bad index");
            if (sp - fr[nfr].base < nargs) trap(64, "stack underflow");
            if (imm >= (int64_t)codelen) trap(75, "jump out of range");
            if (nfr >= 32) trap(74, "call depth exceeded");
            Fr *nf = &fr[nfr + 1]; memset(nf, 0, sizeof *nf);
            for (int k = 0; k < 16; k++) nf->loc[k] = I(0);
            for (int k = 0; k < nargs; k++) nf->loc[k] = stk[sp - nargs + k];
            sp -= nargs; nf->base = sp; nf->ret = npc; nfr++; npc = (uint32_t)imm; break; }
        case RET: {
            a = pop();
            if (nfr == 0) { if (a.t) trap(66, "type error"); fflush(stdout); return (int)(a.v & 0xff); }
            sp = fr[nfr].base; npc = fr[nfr].ret; nfr--; push(a); break; }
        case HALT: { int64_t v = popi(); fflush(stdout); return (int)(v & 0xff); }
        case LOAD: if (imm >= 16) trap(73, "bad index"); push(fr[nfr].loc[imm]); break;
        case STORE: if (imm >= 16) trap(73, "bad index"); fr[nfr].loc[imm] = pop(); break;
        case GLOAD: if (imm >= 64) trap(73, "bad index"); push(glob[imm]); break;
        case GSTORE: if (imm >= 64) trap(73, "bad index"); glob[imm] = pop(); break;
        case NEWARR: { int64_t k = popi(); if (k < 0 || k > 1024) trap(78, "invalid array size"); push(newobj(2, (int)k)); break; }
        case ALOAD: { int64_t i = popi(); a = pop(); if (a.t != 2) trap(66, "type error");
            if (i < 0 || i >= heap[a.v].len) trap(68, "index out of range"); push(I(heap[a.v].a[i])); break; }
        case ASTORE: { int64_t v = popi(); int64_t i = popi(); a = pop(); if (a.t != 2) trap(66, "type error");
            if (i < 0 || i >= heap[a.v].len) trap(68, "index out of range"); heap[a.v].a[i] = v; break; }
        case ALEN: a = pop(); if (a.t != 2) trap(66, "type error"); push(I(heap[a.v].len)); break;
        case SLEN: a = pop(); if (a.t != 1) trap(66, "type error"); push(I(heap[a.v].len)); break;
        case SCAT: { if (sp - fr[nfr].base < 2) trap(64, "stack underflow"); b = pop(); a = pop(); if (a.t != 1 || b.t != 1) trap(66, "type error");
            int l = heap[a.v].len + heap[b.v].len; if (l > 4096) trap(77, "string too long");
            V r = newobj(1, l); memcpy(heap[r.v].s, heap[a.v].s, heap[a.v].len); memcpy(heap[r.v].s + heap[a.v].len, heap[b.v].s, heap[b.v].len); push(r); break; }
        case SIDX: { int64_t i = popi(); a = pop(); if (a.t != 1) trap(66, "type error");
            if (i < 0 || i >= heap[a.v].len) trap(68, "index out of range"); push(I((uint8_t)heap[a.v].s[i])); break; }
        case SSUB: { int64_t l = popi(); int64_t st = popi(); a = pop(); if (a.t != 1) trap(66, "type error");
            if (st < 0 || l < 0 || st + l > heap[a.v].len) trap(68, "index out of range");
            V r = newobj(1, (int)l); memcpy(heap[r.v].s, heap[a.v].s + st, l); push(r); break; }
        case ITOS: { int64_t v = popi(); char tmp[32]; int l = snprintf(tmp, sizeof tmp, "%lld", (long long)v);
            V r = newobj(1, l); memcpy(heap[r.v].s, tmp, l); push(r); break; }
        case STOI: { a = pop(); if (a.t != 1) trap(66, "type error"); char *e; long long v = strtoll(heap[a.v].s, &e, 10);
            if (e == heap[a.v].s) v = 0; push(I(v)); break; }
        case CHR: { int64_t v = popi(); V r = newobj(1, 1); heap[r.v].s[0] = (char)(v & 0xff); push(r); break; }
        case PRINT: { a = pop();
            if (a.t == 0) printf("%lld\n", (long long)a.v);
            else if (a.t == 1) { fwrite(heap[a.v].s, 1, heap[a.v].len, stdout); putchar('\n'); }
            else { putchar('['); for (int k = 0; k < heap[a.v].len; k++) printf(k ? ",%lld" : "%lld", (long long)heap[a.v].a[k]); printf("]\n"); }
            break; }
        case PUTC: { int64_t v = popi(); putchar((int)(v & 0xff)); break; }
        case READ: { char line[4200]; int l = 0, ch, got = 0;
            while ((ch = getchar()) != EOF) { got = 1; if (ch == '\n') break; if (l < 4096) line[l++] = ch; }
            if (!got) push(I(-1)); else { V r = newobj(1, l); memcpy(heap[r.v].s, line, l); push(r); }
            break; }
        case READC: { int ch = getchar(); push(I(ch == EOF ? -1 : ch)); break; }
        case ASSERT: { int64_t v = popi(); if (!v) trap(71, "assertion failed"); break; }
        case DEPTH: push(I(sp - fr[nfr].base)); break;
        case TYPEOF: a = pop(); push(I(a.t)); break;
        }
        pc = npc;
    }
}
