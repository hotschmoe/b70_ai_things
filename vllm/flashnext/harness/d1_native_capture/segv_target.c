#define _POSIX_C_SOURCE 200809L
#include <signal.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>

static void recover(int sig) {
    (void)sig;
    static const char text[] = "D1_RECOVERABLE_HANDLER_RAN\n";
    write(STDOUT_FILENO, text, sizeof(text)-1);
}

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    struct sigaction action = {0};
    action.sa_handler = recover;
    sigemptyset(&action.sa_mask);
    sigaction(SIGSEGV, &action, NULL);
    struct timespec pause = {.tv_sec=0, .tv_nsec=10000000};
    for (int i=0; access(argv[1],F_OK) != 0; ++i) {
        if (i >= 1000) return 3;
        nanosleep(&pause,NULL);
    }
    raise(SIGSEGV);  /* GDB must log then deliver to the existing handler. */
    action.sa_handler = SIG_DFL;
    sigaction(SIGSEGV, &action, NULL);
    raise(SIGSEGV);  /* Preserve actual fatal waitstatus11. */
    return 4;
}
