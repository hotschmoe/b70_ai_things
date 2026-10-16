CONFIG -> Read-only one bounded CPU-leased e41 container, UID1000,
network none, /work mounted read-only, no DRI, no configure/build.

COMMAND -> bin/cpu-run env B70_AGENT=S129-dirty-dep-diagnosis docker run
--rm --network none --cpus 1 -u 1000:1000
-v /mnt/vm_8tb/b70/build/strata-ab:/work:ro --entrypoint bash
sha256:e41d59c0940e1bd2f6f64c9d5d2d7452b77812b0a8407fac8d67cacec5d7fdd2
-c 'source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1;
cd /work/src-ep/build-sycl-ep; ninja -n -d explain strata'

RESULT -> Actual exit0, tool wall9.889389s (container setup included).
Ninja exact extra dirty edge:
output ggml/src/CMakeFiles/ggml-base.dir/ggml.c.o older than most recent input
/work/src-ep/build-sycl-ep/ggml/src/ggml-version.h
(1792024923926870904 vs 1792122090496186302).
Actual ten actions: original four C++ TUs/three archives/main plus one CPU
icx compile ggml.c.o and ggml-base.a archive. No command executed by dryplan.

Current generated header77B, UID1000:1000/mode664:
#pragma once
#define GGML_VERSION "0.24.0"
#define GGML_COMMIT  "3cf03257f"

Source binding: llama-pinned/ggml/CMakeLists.txt:13-40 gets Git short HEAD
and appends -dirty if diff-index reports1. ggml/src/CMakeLists.txt:216 uses
configure_file(ggml-version.h.in ... @ONLY). ggml.c:4 includes it; macro uses
at515/519 return version/commit strings. Actual retained configure.log prints
0.24.0 and3cf03257f. Previous header bytes not retained here: old/new literal
change is UNKNOWN; Ninja identifies the generated header mtime as dependency.
All168 compile edges/rules/existing cache entries had already passed root's
configured guard; this diagnosis did not reconfigure or inhibit dependencies.

Metadata at diagnosis: ggml.c.o272416B UID0:0/mode644/not writable;
ggml-base.a1920010B UID0:0/mode644/not writable. Object parent
CMakeFiles/ggml-base.dir and archive parent ggml/src both UID1000/mode755,
writable. CPU object's .d file absent. Archive command begins cmake -E rm -f,
so parent permits removal/recreation; compiler overwrite may need root's narrow
byte-preserving object replacement. No repair performed by this agent.

VERDICT -> EXACT_ADDITIONAL_GENERATED_HEADER_DEPENDENCY_IDENTIFIED.
The8-action expectation stopped correctly; actual plan10 requires explicit
root disposition. No GPU, compile, cache warm, forced dependency or retry.
