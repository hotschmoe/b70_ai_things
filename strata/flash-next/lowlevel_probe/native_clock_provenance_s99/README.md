# S99 conditional native clock provenance

Small exact records supplement the [S95 profile report](../../../../docs/20261015_strata_current_c2_profile.md)
and [S95 record bundle](../current_c2_profile_s95/README.md).
External source root: `/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/native-clock-provenance-s99-source/`.

CONFIG -> Existing S95 d94e96ca native binary and pinned e41 image;
CPU-only loader/library inspection, no native main, build, new trace or DRI.

COMMAND -> Direct glibc loader `--list --preload` reconstruction and one
selected libstdc++ `steady_clock::now` disassembly. The inspection script is
an exact historical record with absolute external paths; it is not a runnable
in-place controller. `record-copies.json` binds every selected source byte.
`external-source-manifest.json` inventories the complete external audit,
including files deliberately omitted here.

RESULT -> The reconstructed library libstdc++.so.6.0.33, SHA256
`1fd75fe70354a416d75aef22bcae68c47bd25d20e2d0568c30b1a9838cf62f11`,
passes literal clock ID1 to `clock_gettime` and returns nanoseconds. The pinned
header names ID1 `CLOCK_MONOTONIC`; the marker source calls that C++ API.
This establishes the selected library implementation, not the library actually
loaded in historical S95. Native maps/full native LD paths were not retained.

The first inspection incorrectly applied LD_PRELOAD to the ldd shell wrapper;
it attempted profiler initialization and failed255 without /dev/dri. It did
not run native main. The corrected direct-loader inspection completed0.
**Do not replay the failed `inspect-offline.sh`.** That script and its failed
loader output remain external only; the exact failure/correction is disclosed
in `clock-provenance.json` and `findings.txt`.

VERDICT -> CONDITIONAL; historical S95 native-to-trace joins remain unauthorized.
Future provenance can retain native maps/resolved library hashes and loader
controls during an already planned run. Fresh clock brackets still require
uncertainty/drift review. No cross-card clock join, causal idle/wait removal,
physical traffic, speed or serving qualification follows. No runtime test was
performed by this source audit; other experiment/raw records remain external.
