/* Reference server for task re-proto. NOT visible to the agent (only the stripped binary is). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <signal.h>
#include <poll.h>
#include <errno.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#define MAXPAY 512
#define MAXKEYS 64
#define MAXCONN 256
enum { E_BADFRAME = 1, E_BADVER, E_NOAUTH, E_NOTFOUND, E_WRONGTYPE, E_CONFLICT, E_TOOBIG, E_STATE, E_UNKNOWN, E_BADARG, E_PERM, E_FULL, E_AUTHFAIL };
static const char *EMSG[] = {"", "bad frame", "unsupported version", "authentication required", "not found", "wrong type",
                             "transaction conflict", "payload too large", "invalid state", "unknown operation", "bad argument",
                             "permission denied", "store full", "authentication failed"};

typedef struct { uint8_t used, type, klen; char key[33]; uint16_t vlen; uint8_t val[420]; uint64_t exp; } Ent;
typedef struct { Ent e[MAXKEYS]; } Store;
typedef struct {
    int fd, used, id, authed, role, fails, closing, intx, nw, ntouch;
    uint8_t in[2048]; int inlen;
    char watch[8][34];
    Store base, work;
    char touched[MAXKEYS][33];
} Conn;

static Store G;
static uint64_t CLOCK = 0, SEED = 1;
static Conn *conns[MAXCONN];
static int accepted = 0;

static const struct { const char *user, *secret; int role; } USERS[] = {
    {"alice", "wonderland", 3}, {"bob", "builder99", 2}, {"carol", "readonly!", 1}, {"dave", "d4v3-r0cks", 2}};

static uint64_t splitmix(uint64_t z) {
    z += 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static uint64_t rotl(uint64_t x, int r) { return (x << r) | (x >> (64 - r)); }
static uint64_t auth_hash(const char *secret, const uint8_t *nonce, const uint8_t *user, int ul) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (const char *p = secret; *p; p++) { h ^= (uint8_t)*p; h *= 0x100000001b3ULL; }
    for (int i = 0; i < 8; i++) { h ^= nonce[i]; h *= 0x100000001b3ULL; h = rotl(h, 5); }
    for (int i = 0; i < ul; i++) { h ^= user[i]; h *= 0x100000001b3ULL; }
    h ^= h >> 29; h *= 0xBF58476D1CE4E5B9ULL; h ^= h >> 32;
    return h;
}

static void send_frame(Conn *c, uint8_t type, uint32_t rid, const uint8_t *p, int n) {
    uint8_t buf[MAXPAY + 64];
    buf[0] = 'K'; buf[1] = 'V'; buf[2] = 2; buf[3] = type;
    for (int i = 0; i < 4; i++) buf[4 + i] = rid >> (8 * i);
    buf[8] = n & 0xff; buf[9] = n >> 8;
    memcpy(buf + 10, p, n);
    int off = 0, tot = 10 + n;
    while (off < tot) { ssize_t w = send(c->fd, buf + off, tot - off, MSG_NOSIGNAL); if (w <= 0) { c->closing = 1; return; } off += w; }
}
static void send_err(Conn *c, uint32_t rid, int code) {
    uint8_t b[64]; b[0] = code; int l = strlen(EMSG[code]); memcpy(b + 1, EMSG[code], l);
    send_frame(c, 0xFF, rid, b, 1 + l);
}

static Ent *find(Store *s, const char *k, int kl) {
    for (int i = 0; i < MAXKEYS; i++) if (s->e[i].used && s->e[i].klen == kl && !memcmp(s->e[i].key, k, kl)) return &s->e[i];
    return NULL;
}
static int count(Store *s) { int n = 0; for (int i = 0; i < MAXKEYS; i++) n += s->e[i].used; return n; }
static Ent *alloc(Store *s, const char *k, int kl) {
    for (int i = 0; i < MAXKEYS; i++) if (!s->e[i].used) { memset(&s->e[i], 0, sizeof(Ent)); s->e[i].used = 1; s->e[i].klen = kl; memcpy(s->e[i].key, k, kl); return &s->e[i]; }
    return NULL;
}
static int ent_eq(Ent *a, Ent *b) {
    if (!a || !b) return a == b;
    return a->type == b->type && a->vlen == b->vlen && a->exp == b->exp && !memcmp(a->val, b->val, a->vlen);
}

static int match(const char *pat, const char *k, int kl) {
    int pl = strlen(pat);
    if (pl && pat[pl - 1] == '*') return kl >= pl - 1 && !memcmp(pat, k, pl - 1);
    return kl == pl && !memcmp(pat, k, kl);
}
static void notify(Conn *actor, int ev, const char *k, int kl) {
    uint8_t b[40]; b[0] = ev; b[1] = kl; memcpy(b + 2, k, kl);
    for (int i = 0; i < MAXCONN; i++) {
        Conn *c = conns[i];
        if (!c || !c->used || c == actor || c->closing) continue;
        for (int w = 0; w < c->nw; w++) if (match(c->watch[w], k, kl)) { send_frame(c, 0xA0, 0, b, 2 + kl); break; }
    }
}

typedef struct { const uint8_t *p; int n, bad; } Cur;
static uint64_t take(Cur *c, int w) {
    if (c->n < w) { c->bad = 1; return 0; }
    uint64_t v = 0; for (int i = 0; i < w; i++) v |= (uint64_t)c->p[i] << (8 * i);
    c->p += w; c->n -= w; return v;
}
static const uint8_t *takeb(Cur *c, int w) {
    if (c->n < w) { c->bad = 1; return NULL; }
    const uint8_t *r = c->p; c->p += w; c->n -= w; return r;
}

/* value validation: returns stored length or -1 */
static int check_val(int type, Cur *c, const uint8_t **v) {
    *v = c->p;
    if (type == 0) { if (c->n != 8) return -1; return 8; }
    if (type == 1) { if (c->n > 400) return -1; return c->n; }
    if (type == 2) {
        if (c->n < 1) return -1;
        int cnt = c->p[0], o = 1;
        if (cnt > 16) return -1;
        for (int i = 0; i < cnt; i++) { if (o >= c->n) return -1; o += 1 + c->p[o]; if (o > c->n) return -1; }
        if (o != c->n || o > 400) return -1;
        return o;
    }
    return -1;
}
static int put_record(uint8_t *out, Ent *e) {
    uint32_t ttl = e->exp ? (uint32_t)(e->exp - CLOCK) : 0;
    out[0] = e->type; for (int i = 0; i < 4; i++) out[1 + i] = ttl >> (8 * i);
    out[5] = e->vlen & 0xff; out[6] = e->vlen >> 8; memcpy(out + 7, e->val, e->vlen);
    return 7 + e->vlen;
}

static int is_write(int op) { return op == 3 || op == 5 || op == 6 || op == 17; }
static void touch(Conn *c, const char *k, int kl) {
    for (int i = 0; i < c->ntouch; i++) if (strlen(c->touched[i]) == (size_t)kl && !memcmp(c->touched[i], k, kl)) return;
    memcpy(c->touched[c->ntouch], k, kl); c->touched[c->ntouch][kl] = 0; c->ntouch++;
}
static int cmpstr(const void *a, const void *b) { return strcmp((const char *)a, (const char *)b); }

static void handle(Conn *c, int op, uint32_t rid, const uint8_t *pay, int n) {
    uint8_t out[MAXPAY + 64];
    Cur cu = {pay, n, 0};
    if (!c->authed && op != 1 && op != 2) { send_err(c, rid, E_NOAUTH); return; }
    if (op < 1 || op > 18) { send_err(c, rid, E_UNKNOWN); return; }
    if (c->authed) {
        int ok = 1;
        if (c->role == 1) ok = !(is_write(op) || op == 7);
        if (c->role == 2) ok = op != 7;
        if (!ok) { send_err(c, rid, E_PERM); return; }
    }
    Store *S = c->intx ? &c->work : &G;
    switch (op) {
    case 1: send_frame(c, 0x81, rid, pay, n); return;
    case 2: {
        if (c->authed) { send_err(c, rid, E_STATE); return; }
        int ul = take(&cu, 1); const uint8_t *u = takeb(&cu, ul); uint64_t h = take(&cu, 8);
        if (cu.bad || cu.n) { send_err(c, rid, E_BADARG); return; }
        uint8_t nonce[8]; uint64_t z = splitmix(SEED + (uint64_t)c->id * 0x9E3779B97F4A7C15ULL);
        for (int i = 0; i < 8; i++) nonce[i] = z >> (8 * i);
        int found = -1;
        for (unsigned i = 0; i < 4; i++) if ((int)strlen(USERS[i].user) == ul && !memcmp(USERS[i].user, u, ul)) found = i;
        if (found < 0 || auth_hash(USERS[found].secret, nonce, u, ul) != h) {
            send_err(c, rid, E_AUTHFAIL);
            if (++c->fails >= 3) c->closing = 1;
            return;
        }
        c->authed = 1; c->role = USERS[found].role; out[0] = c->role; send_frame(c, 0x82, rid, out, 1); return;
    }
    case 3: {
        int kl = take(&cu, 1); const uint8_t *k = takeb(&cu, kl); int type = take(&cu, 1); uint32_t ttl = take(&cu, 4);
        if (cu.bad || kl < 1 || kl > 32 || memchr(k, 0, kl)) { send_err(c, rid, E_BADARG); return; }
        const uint8_t *v; int vl = check_val(type, &cu, &v);
        if (vl < 0) { send_err(c, rid, E_BADARG); return; }
        Ent *e = find(S, (const char *)k, kl);
        if (!e) { if (count(S) >= MAXKEYS) { send_err(c, rid, E_FULL); return; } e = alloc(S, (const char *)k, kl); }
        e->type = type; e->vlen = vl; memcpy(e->val, v, vl); e->exp = ttl ? CLOCK + ttl : 0;
        if (c->intx) touch(c, (const char *)k, kl); else notify(c, 1, (const char *)k, kl);
        send_frame(c, 0x83, rid, NULL, 0); return;
    }
    case 4: {
        int kl = take(&cu, 1); const uint8_t *k = takeb(&cu, kl);
        if (cu.bad || cu.n || kl < 1 || memchr(k, 0, kl)) { send_err(c, rid, E_BADARG); return; }
        if (c->intx) touch(c, (const char *)k, kl);
        Ent *e = find(S, (const char *)k, kl);
        if (!e) { send_err(c, rid, E_NOTFOUND); return; }
        send_frame(c, 0x84, rid, out, put_record(out, e)); return;
    }
    case 5: {
        int kl = take(&cu, 1); const uint8_t *k = takeb(&cu, kl);
        if (cu.bad || cu.n || kl < 1 || memchr(k, 0, kl)) { send_err(c, rid, E_BADARG); return; }
        Ent *e = find(S, (const char *)k, kl);
        if (c->intx) touch(c, (const char *)k, kl);
        if (e) { e->used = 0; if (!c->intx) notify(c, 2, (const char *)k, kl); }
        out[0] = e ? 1 : 0; send_frame(c, 0x85, rid, out, 1); return;
    }
    case 6: {
        int kl = take(&cu, 1); const uint8_t *k = takeb(&cu, kl); int64_t d = (int64_t)take(&cu, 8);
        if (cu.bad || cu.n || kl < 1 || kl > 32 || memchr(k, 0, kl)) { send_err(c, rid, E_BADARG); return; }
        Ent *e = find(S, (const char *)k, kl);
        if (e && e->type != 0) { send_err(c, rid, E_WRONGTYPE); return; }
        if (!e) { if (count(S) >= MAXKEYS) { send_err(c, rid, E_FULL); return; } e = alloc(S, (const char *)k, kl); e->type = 0; e->vlen = 8; memset(e->val, 0, 8); }
        uint64_t cur = 0; for (int i = 0; i < 8; i++) cur |= (uint64_t)e->val[i] << (8 * i);
        cur += (uint64_t)d;
        for (int i = 0; i < 8; i++) e->val[i] = cur >> (8 * i);
        if (c->intx) touch(c, (const char *)k, kl); else notify(c, 4, (const char *)k, kl);
        for (int i = 0; i < 8; i++) out[i] = cur >> (8 * i);
        send_frame(c, 0x86, rid, out, 8); return;
    }
    case 7: {
        uint32_t t = take(&cu, 4);
        if (cu.bad || cu.n || t == 0 || t > 100000) { send_err(c, rid, E_BADARG); return; }
        if (c->intx) { send_err(c, rid, E_STATE); return; }
        CLOCK += t;
        char dead[MAXKEYS][33]; int nd = 0;
        for (int i = 0; i < MAXKEYS; i++) if (G.e[i].used && G.e[i].exp && G.e[i].exp <= CLOCK) {
            memcpy(dead[nd], G.e[i].key, G.e[i].klen); dead[nd][G.e[i].klen] = 0; nd++; G.e[i].used = 0; }
        qsort(dead, nd, 33, cmpstr);
        for (int i = 0; i < nd; i++) notify(c, 3, dead[i], strlen(dead[i]));
        for (int i = 0; i < 8; i++) out[i] = CLOCK >> (8 * i);
        out[8] = nd & 0xff; out[9] = nd >> 8; send_frame(c, 0x87, rid, out, 10); return;
    }
    case 8:
        if (n) { send_err(c, rid, E_BADARG); return; }
        if (c->intx) { send_err(c, rid, E_STATE); return; }
        c->intx = 1; c->base = G; c->work = G; c->ntouch = 0; send_frame(c, 0x88, rid, NULL, 0); return;
    case 9: {
        if (n) { send_err(c, rid, E_BADARG); return; }
        if (!c->intx) { send_err(c, rid, E_STATE); return; }
        c->intx = 0;
        for (int i = 0; i < c->ntouch; i++) {
            int kl = strlen(c->touched[i]);
            if (!ent_eq(find(&c->base, c->touched[i], kl), find(&G, c->touched[i], kl))) { send_err(c, rid, E_CONFLICT); return; }
        }
        char ks[MAXKEYS][33]; int nk = c->ntouch;
        for (int i = 0; i < nk; i++) strcpy(ks[i], c->touched[i]);
        qsort(ks, nk, 33, cmpstr);
        int applied = 0;
        for (int i = 0; i < nk; i++) {
            int kl = strlen(ks[i]);
            Ent *b = find(&c->base, ks[i], kl), *w = find(&c->work, ks[i], kl), *g = find(&G, ks[i], kl);
            if (ent_eq(b, w)) continue;
            if (w) { if (!g) g = alloc(&G, ks[i], kl); g->type = w->type; g->vlen = w->vlen; memcpy(g->val, w->val, w->vlen); g->exp = w->exp; notify(c, 1, ks[i], kl); }
            else if (g) { g->used = 0; notify(c, 2, ks[i], kl); }
            applied++;
        }
        out[0] = applied & 0xff; out[1] = applied >> 8; send_frame(c, 0x89, rid, out, 2); return;
    }
    case 10:
        if (n) { send_err(c, rid, E_BADARG); return; }
        if (!c->intx) { send_err(c, rid, E_STATE); return; }
        c->intx = 0; send_frame(c, 0x8A, rid, NULL, 0); return;
    case 11: case 12: {
        int kl = take(&cu, 1); const uint8_t *k = takeb(&cu, kl);
        if (cu.bad || cu.n || kl < 1 || kl > 32 || memchr(k, 0, kl)) { send_err(c, rid, E_BADARG); return; }
        char pat[34]; memcpy(pat, k, kl); pat[kl] = 0;
        int idx = -1; for (int i = 0; i < c->nw; i++) if (!strcmp(c->watch[i], pat)) idx = i;
        if (op == 11) {
            if (idx < 0) { if (c->nw >= 8) { send_err(c, rid, E_FULL); return; } strcpy(c->watch[c->nw++], pat); }
        } else {
            if (idx < 0) { send_err(c, rid, E_NOTFOUND); return; }
            for (int i = idx; i < c->nw - 1; i++) strcpy(c->watch[i], c->watch[i + 1]);
            c->nw--;
        }
        send_frame(c, 0x80 | op, rid, NULL, 0); return;
    }
    case 13: {
        int kl = take(&cu, 1); const uint8_t *k = takeb(&cu, kl);
        if (cu.bad || cu.n || kl > 32 || memchr(k, 0, kl)) { send_err(c, rid, E_BADARG); return; }
        char pat[34]; memcpy(pat, k, kl); pat[kl] = 0;
        char ks[MAXKEYS][33]; int nk = 0;
        for (int i = 0; i < MAXKEYS; i++) if (S->e[i].used && (kl == 0 || match(pat, S->e[i].key, S->e[i].klen))) {
            memcpy(ks[nk], S->e[i].key, S->e[i].klen); ks[nk][S->e[i].klen] = 0; nk++; }
        qsort(ks, nk, 33, cmpstr);
        int o = 2; out[0] = nk & 0xff; out[1] = nk >> 8;
        for (int i = 0; i < nk; i++) { int l = strlen(ks[i]); out[o++] = l; memcpy(out + o, ks[i], l); o += l; }
        send_frame(c, 0x8D, rid, out, o); return;
    }
    case 14: {
        if (n) { send_err(c, rid, E_BADARG); return; }
        char t[160]; int l = snprintf(t, sizeof t, "clock=%llu keys=%d accepted=%d watches=%d tx=%d\n", (unsigned long long)CLOCK, count(S), accepted, c->nw, c->intx);
        send_frame(c, 0x8E, rid, (uint8_t *)t, l); return;
    }
    case 15: send_frame(c, 0x8F, rid, NULL, 0); c->closing = 1; return;
    case 16: {
        int cnt = take(&cu, 1);
        if (cu.bad || cnt < 1 || cnt > 8) { send_err(c, rid, E_BADARG); return; }
        int o = 1; out[0] = cnt;
        for (int i = 0; i < cnt; i++) {
            int kl = take(&cu, 1); const uint8_t *k = takeb(&cu, kl);
            if (cu.bad || kl < 1 || memchr(k, 0, kl)) { send_err(c, rid, E_BADARG); return; }
            if (c->intx) touch(c, (const char *)k, kl);
            Ent *e = find(S, (const char *)k, kl);
            if (!e) out[o++] = 0; else { out[o++] = 1; o += put_record(out + o, e); }
            if (o > MAXPAY) { send_err(c, rid, E_TOOBIG); return; }
        }
        if (cu.n) { send_err(c, rid, E_BADARG); return; }
        send_frame(c, 0x90, rid, out, o);
        return;
    }
    case 17: {
        int kl = take(&cu, 1); const uint8_t *k = takeb(&cu, kl); uint32_t ttl = take(&cu, 4);
        if (cu.bad || cu.n || kl < 1 || memchr(k, 0, kl)) { send_err(c, rid, E_BADARG); return; }
        if (c->intx) touch(c, (const char *)k, kl);
        Ent *e = find(S, (const char *)k, kl);
        if (e) { e->exp = ttl ? CLOCK + ttl : 0; if (!c->intx) notify(c, 1, (const char *)k, kl); }
        out[0] = e ? 1 : 0; send_frame(c, 0x91, rid, out, 1); return;
    }
    case 18: {
        int kl = take(&cu, 1); const uint8_t *k = takeb(&cu, kl);
        if (cu.bad || cu.n || kl < 1 || memchr(k, 0, kl)) { send_err(c, rid, E_BADARG); return; }
        if (c->intx) touch(c, (const char *)k, kl);
        Ent *e = find(S, (const char *)k, kl);
        if (!e) { send_err(c, rid, E_NOTFOUND); return; }
        uint32_t t = e->exp ? (uint32_t)(e->exp - CLOCK) : 0xFFFFFFFFu;
        for (int i = 0; i < 4; i++) out[i] = t >> (8 * i);
        send_frame(c, 0x92, rid, out, 4); return;
    }
    }
}

static void drain(Conn *c) {
    while (c->inlen >= 10 && !c->closing) {
        uint8_t *h = c->in;
        uint32_t rid = h[4] | (h[5] << 8) | (h[6] << 16) | ((uint32_t)h[7] << 24);
        int len = h[8] | (h[9] << 8);
        if (h[0] != 'K' || h[1] != 'V') { send_err(c, 0, E_BADFRAME); c->closing = 1; return; }
        if (h[2] != 2) { send_err(c, rid, E_BADVER); c->closing = 1; return; }
        if (len > MAXPAY) { send_err(c, rid, E_TOOBIG); c->closing = 1; return; }
        if (c->inlen < 10 + len) return;
        uint8_t pay[MAXPAY + 1]; memcpy(pay, h + 10, len);
        int op = h[3];
        memmove(c->in, c->in + 10 + len, c->inlen - 10 - len); c->inlen -= 10 + len;
        handle(c, op, rid, pay, len);
    }
}

int main(int argc, char **argv) {
    int port = 0;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "--port")) port = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--seed")) SEED = strtoull(argv[i + 1], NULL, 10);
    }
    if (!port) { fprintf(stderr, "usage: kvd --port N [--seed S]\n"); return 2; }
    signal(SIGPIPE, SIG_IGN);
    int ls = socket(AF_INET, SOCK_STREAM, 0), one = 1;
    setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    struct sockaddr_in a = {0}; a.sin_family = AF_INET; a.sin_port = htons(port); a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (bind(ls, (struct sockaddr *)&a, sizeof a) || listen(ls, 16)) { perror("kvd"); return 1; }
    for (;;) {
        struct pollfd pf[MAXCONN + 1]; int idx[MAXCONN + 1], np = 0;
        pf[np].fd = ls; pf[np].events = POLLIN; idx[np++] = -1;
        for (int i = 0; i < MAXCONN; i++) if (conns[i] && conns[i]->used) { pf[np].fd = conns[i]->fd; pf[np].events = POLLIN; idx[np++] = i; }
        if (poll(pf, np, -1) < 0) { if (errno == EINTR) continue; return 1; }
        for (int k = 0; k < np; k++) {
            if (!(pf[k].revents & (POLLIN | POLLHUP | POLLERR))) continue;
            if (idx[k] < 0) {
                int fd = accept(ls, NULL, NULL); if (fd < 0 || accepted + 1 >= MAXCONN) { if (fd >= 0) close(fd); continue; }
                Conn *c = calloc(1, sizeof *c); c->fd = fd; c->used = 1; c->id = ++accepted; conns[c->id] = c;
                uint8_t b[16]; uint64_t z = splitmix(SEED + (uint64_t)c->id * 0x9E3779B97F4A7C15ULL);
                for (int i = 0; i < 8; i++) b[i] = z >> (8 * i);
                b[8] = 5; memcpy(b + 9, "kvd/2", 5);
                send_frame(c, 0x80, 0, b, 14);
                continue;
            }
            Conn *c = conns[idx[k]];
            ssize_t r = read(c->fd, c->in + c->inlen, sizeof c->in - c->inlen);
            if (r <= 0) { close(c->fd); c->used = 0; continue; }
            if (c->closing) { c->inlen = 0; continue; }   /* lingering: discard until the peer closes */
            c->inlen += r;
            drain(c);
            if (c->closing) { shutdown(c->fd, SHUT_WR); c->inlen = 0; }
        }
    }
}
