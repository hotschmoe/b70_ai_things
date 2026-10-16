# Runtime queue/readiness lead: source and official documentation only

CONFIG -> Pinned e41 oneAPI2026.1.0-235 (compiler2026.1.0.20260617),
libsycl.so9.0.0; both LevelZero adapter DSOs0.12.0 installed. Current driver
baseline is ComputeRuntime26.22.38646.4 per project evidence; this task does
not query GPU or change the driver. Actual active adapter remains UNKNOWN
until separate S140 discovery; loading/mapping BOTH DSOs proves neither active.

COMMAND -> Bounded local source reads; no-DRI/network-none pinned-image SDK
package/list/string reads and one explicit installed helper disassembly; primary
Intel/UR web documentation and SHA-pinned current upstream source. No GPU,
main build, model/trace reads or runtime setting changes. Docker reads took
about10s per startup. First marker inspection lacked container rg (exit127);
next bounded read used grep. No failed GPU or experimental runtime attempt.
Several pre-package GitHub paths returned404; no exactSDK-source identity was
inferred. The preserved current upstream SHA is b29c1834fc5795c9748173a1b3ab48e3d75abd28
(2026-10-15), later than the installed June package; NOT byteidentity proof.

RESULT -> A real profiling-enabled construction seam exists, but an unset->0
UR_L0_USE_DRIVER_INORDER_LISTS benchmark could be a complete no-op.

1. mtp.cpp/verify.cpp both define DPCT_PROFILING_ENABLED atL2 before dpct.
   dpct/device.hpp create_queue_impl L675-690 adds enable_profiling under this
   macro, while create_queue selects in_order L616-627. MTP load L517/527
   uses this factory. This is independent of verifier's optional prof_on_
   (STRATA_VERIFY_PROFILE atL633). Source condition is clear; actual main UR
   queue flags are not captured here and header/inlining selection is not
   certified by a source read. Do not remove the macro globally: sync_barrier
   L989-994 also selects a barrier versus a dummy kernel under the same macro.

2. Intel's 2025 notes document driver-inorder overhead for profiling-enabled
   queues and suggest UR_L0_USE_DRIVER_INORDER_LISTS=0. This is historical
   motivation, NOT proof of the installed2026.1 V2 path. The 2026 notes state
   V2 defaults on Xe2/Battlemage. Current compiler documentation likewise
   selects V2 by platform and describes SYCL_UR_USE_LEVEL_ZERO_V2 overrides.
   [2025 notes](https://www.intel.com/content/www/us/en/developer/articles/release-notes/oneapi-dpcpp/2025.html)
   [2026 notes](https://www.intel.com/content/www/us/en/developer/articles/release-notes/oneapi-dpcpp/2026.html)
   [DPC++ environment](https://intel.github.io/llvm/EnvironmentVariables.html)

3. Current official upstream common/adapter.cpp L307-337 chooses V2 for GPU
   IP >=0x05004000 unless an explicit adapter override exists. Both adapter
   entrypoints log a version-specific SKIP before exposing that adapter.
   Installed e41 strings contain EXACT `Skipping L0 V1 adapter: {}` and V2
   equivalent, and the default-IP reason. This supports a tiny actual-discovery
   test, not a source-only declaration of the selected runtime.

4. Current upstream V2 queue_create.cpp uses the in-order V2 queue class; its
   constructor requests driver in-order directly from the command-list cache.
   That inspected queue path does not consult UR_L0_USE_DRIVER_INORDER_LISTS.
   The variable's string in BOTH DSOs originates in the sharedplatformhelper;
   string presence is not proof of use by a V2 queue. Current source is an
   interpretation guide, not an exact June-package callgraph certificate.

5. Installed legacy helper allowDriverInOrderLists(bool) at0x1409c0 actually
   parses the environment once and caches it. At0x1409ec-1409f2 the bool input
   chooses explicit-env versus compatible-driver state. At0x140a55-140a6f it
   tests driver >=1.6.32149 and ORs the env bit. Thus OnlyIfRequested=true
   gives false for unset AND0; false returns compatible-driver ORenv. A bare
   0 is not a universal override of the compatibility-derived boolean. Current
   upstream legacy immediate/regular queue paths pass true (queue.cppL2397,
   L2548), so their unset and0 are equal. This is not universal SDKV1/V2
   compiled-callsite proof. Do not equate package26.22 numbering with the
   helper's LevelZero1.6 build tuple without actual driver metadata.

ONE next finite diagnostic -> S140 reuses actual S135 mode0 executable with
only discovery logging. Exactlyone default-IP skip + other-adapter loaded +
original nativeLevelZero assertion/strict8depthA-B-A qualifies active version;
ambiguous/missing logs HOLD. No forcedV1 or env sweep. Source/full lifecycle
prep is separately frozen at ../ur-adapter-discovery-prep. S135's explicit
queues have ONLY in_order, so discovery cannot measure mainqueueprofiling.
Official logger syntax:
https://oneapi-src.github.io/unified-runtime/core/INTRO.html

After actual adapter discovery, ONE useful GPU candidate is a private
runtimeenable_profiling0/1 graph submission comparison on the SAME adapter,
SAME executable and exactintegerknown kernels, with no unitrace/preload/loggers
in timed windows. This avoids a likely no-op legacy environment toggle and
requires only a small private adapter build. Keep hostsubmitreturn separately
from wholechaincompletion and verify freshmapped outputs/guards. No main
rebuild until this discriminator has an effect. Exact source proposal follows
separately if root selects it; this source audit authorizes no execution.

VERDICT -> Worth a privateprofiling-property discriminator IF S140 establishes
actual adapter. NOT justified: blindly forceV1, blindly unset-vs0 driverlists,
remove synchronization, delete all profiling macros, or attribute S136reset
service/uncovered intervals to CPU overhead/physicalidle. No forecast or claim
that a runtime setting fixed readiness. Any runtime gain would be wholeadapter
configuration behavior until individually attributed.
