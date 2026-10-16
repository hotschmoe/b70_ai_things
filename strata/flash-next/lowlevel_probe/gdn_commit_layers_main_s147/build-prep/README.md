# S147 compile-only derivative of completed S143 main build

CONFIG -> SOURCE ONLY, no apply/configure/Ninja/build/GPU. Root reviews/applies exact S147V2 source first and owns actual Opus/Fable decisions. Same e41/noDRI/netnone/UID1000/cpu-run4jobs/diskTMP/stable owned offlinecache. Existing configured compile/link options byte-pinned unchanged; no configure, optional host/ISA/marker certification, old main/archive backup/hash, model scan or cache warm.

COMMAND -> Root only after source apply, six required postapply SHA arguments:

bash build.sh 66726a2abf0f2374ece12bd802ef4ebdad40bada27290a7afb480c849063cb23 452a051e3a2219d0c6a00c8cdf795b798e763fcc793fbbb4dff1b4dda7903aff e6314c05da00d175ae2990a5157bc24d3bea16602afa2138858eaee77200dbdd dd08daf548eab9ab69375ed1702dfa85214bd1431e9957126d12eea51ed569d1 f669057242387e85ba3d433e8be0acacfff0c835a28af34f60fc278ccd88410b 929813472cd1ee9710c811e9c05fe48e8e459daa9fb62527f4cedb0839b86afb

RESULT -> PENDING. Actual dryplan guarded EXACTLY FIVE actions: verify.cpp.o +verify_kernels.dp.cpp.o compiles (order unrestricted), libstrata_engine.a +libstrata_kernels.a archives, one strata link. O3/precise/SG32/per-kernel/backend/div-sqrt/cache flags retained. Any unexpected dependency/action stops BEFORE compilation for root diagnosis; no force/inhibit/retry. No assumed five-action PASS. Existing S143 controller adapted only affected-source/target identities and counts; review delta-from-s143.patch. Ownership/stat check includes changedobjects/archives without old hashes. New output build/strata-s147 retained via reflink; only newmain/two changedobjects/two changedarchives hashed once, plus source/build-system receipts. Source/build-system pre/post exact; cache counters descriptive/UNKNOWN ifunparsed, no warm.

VERDICT -> Root execution pending. Default-OFF production path must still qualify fresh sameNEWbinary OFF/ON serving with output/work/capture/marker/functionals/normalhealth. All other experimental flags0, including parked QSAgroup. S145 private wholebundle gain is not a serving prediction. No extra certification framework added.
