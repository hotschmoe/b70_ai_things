# S143 compile-only main derivative of completed S133 build

CONFIG -> Root applies ONLY reviewed verify.cpp source patch c28de288... after full source read and S142 full closure. Existing CMake/cache/precise/SG32/BMG-G31/frontend options unchanged; no configure. Same e41 image, UID1000, noDRI/netnone, four CPU jobs, disk TMP, owned existing stable offlinecache. Native/mtp/CMake sources remain their current exact SHA pins. No old main/object/archive backup/hash, no model/cachewarm/host-ISA certification.

COMMAND -> Root only, AFTER source apply:

bash build.sh 66726a2abf0f2374ece12bd802ef4ebdad40bada27290a7afb480c849063cb23 452a051e3a2219d0c6a00c8cdf795b798e763fcc793fbbb4dff1b4dda7903aff e6314c05da00d175ae2990a5157bc24d3bea16602afa2138858eaee77200dbdd dd08daf548eab9ab69375ed1702dfa85214bd1431e9957126d12eea51ed569d1 c28de288061beae19949982ff3de7cf0a54fcae4d23f61e3a0374f90aeb435c6

RESULT -> PENDING; prep executed no Ninja/configure/build. Existing S133 controller adapted only changed source/target from nativeMMVQ/kernel archive to verifyTU/engine archive, five postapply SHA arguments, current native marker-fix SHA, new output strata-s143. Actual dryplan must contain exactly:

1. CMakeFiles/strata_engine.dir/src/core/verify.cpp.o compile (O3 precise/SG32/per-kernel options retained).
2. libstrata_engine.a archive.
3. strata final link with same backend options and existing offlinecache.

Unexpected action/dependency stops BEFORE compilation; no forced dependency inhibition or automatic retry. Copied exact current configured-system bytepins guard flags/cache, no speculative three-action PASS. Stable cache counters before/after are descriptive only; missing/unparsed stats stay UNKNOWN. New main,changed verify object,changed engine archive hashed ONCE after build; source/build-system pre/post SHA gates. Freeze new executable build/strata-s143 for fresh samebinary S144control/ON. No extra GPU touch.

VERDICT -> Root review/execution pending. Minimal derivative delta is delta-from-s133.patch; no new build framework or mandatory extra model/certification round. Root already accepted actual-model conditional main sourceGO after S141 strict privatePASS; runtime serving still must satisfy text/work/capture/marker/normal teardown/health gates.
