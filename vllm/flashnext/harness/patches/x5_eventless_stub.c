#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>

/* Keep Torch and SYCL out of the preload dependency graph. The helper is
 * loaded explicitly after Python imports Torch. Preserve errors: a missing
 * helper aborts instead of silently reverting to a different replay path. */
typedef void (*replay_fn)(void*);
static _Atomic(replay_fn) target;

void x5_replay(void*) __asm__("_ZN2at3xpu12XPUGraphImpl6replayEv");
void x5_replay(void* self) {
    replay_fn fn = atomic_load_explicit(&target, memory_order_acquire);
    if (!fn) {
        fn = (replay_fn)dlsym(RTLD_DEFAULT, "b70_x5_eventless_impl");
        if (!fn) {
            fprintf(stderr, "X5_LAZY_EVENTLESS_HELPER_MISSING: %s\n", dlerror());
            abort();
        }
        atomic_store_explicit(&target, fn, memory_order_release);
        fprintf(stderr, "X5_LAZY_EVENTLESS_STUB_ACTIVE\n");
    }
    fn(self);
}
