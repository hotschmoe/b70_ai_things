# Source47 grouped metadata allocation candidate

CONFIG -> Preserve actual45 source/build/model failure. Complete local Info
was allocated by make_shared on ordinary host heap, including a/b/c matrix
pointer arrays. Intel's oneMKL USM interface requires vector/matrix parameters
to reference device-accessible USM. This allocation risk is concrete; the
observed null GPU fault's cause remains unconfirmed.

Primary contract sources:
- https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2025-0/gemm-batch.html
- https://cdrdv2-public.intel.com/834387/onemkl_developer-reference-dpcpp_2025.0-772045-834387.pdf

COMMAND -> New patch allocates the entire Info via malloc_host<Info>(1,q),
placement value-initializes it, and uses a shared_ptr with context-only custom
deleter. The ordinary-heap control block is never passed to BLAS. The shared
owner is captured by the host_task depending on the returned MKL event. The
event gates lifetime; it does not itself own the allocation. On submission
failure the local shared owner remains through the original failure-only
queue retirement. Ambiguous accepted callbacks retain their own shared owner,
avoiding manual double free. sycl::free uses captured context, not the queue.

RESULT -> Exact one-file apply/reverse passed. GEMM call/strides/types/math
are byte-identical. Normal explicit waits and GPU queries are unchanged.
No compiler, GPU, model or provider execution by source author.

VERDICT -> Fresh source47/all8 build and bounded high-ne16..64 single/two-queue
fixture are required before any model trial. Deallocation inside host_task is
an actual runtime/lifetime risk to test; no safety, cause or performance claim
is transferred from source45 or the prior six small operator cases.
