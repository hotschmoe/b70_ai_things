/* segv_bt.c -- LD_PRELOAD SIGSEGV/SIGABRT native-backtrace printer for the breakable-capture
 * segfault hunt (JOURNAL 2026-07-02). backtrace_symbols_fd to stderr, then default action. */
#define _GNU_SOURCE
#include <execinfo.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

static void handler(int sig, siginfo_t *si, void *uc) {
    void *bt[64];
    int n = backtrace(bt, 64);
    dprintf(2, "\n=== segv_bt: signal %d at addr %p, %d frames ===\n", sig, si ? si->si_addr : 0, n);
    backtrace_symbols_fd(bt, n, 2);
    dprintf(2, "=== segv_bt end ===\n");
    signal(sig, SIG_DFL);
    raise(sig);
}

__attribute__((constructor)) static void install(void) {
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_sigaction = handler;
    sa.sa_flags = SA_SIGINFO | SA_RESETHAND;
    sigaction(SIGSEGV, &sa, NULL);
    sigaction(SIGABRT, &sa, NULL);
    sigaction(SIGBUS, &sa, NULL);
}
