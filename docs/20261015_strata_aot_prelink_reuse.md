# Strata AOT prelink reuse, 2026-10-15

CONFIG -> COMMAND -> RESULT -> VERDICT.

## CONFIG: remove repeated compilation of unchanged device images

Small oracle adapters were relinking unchanged native MMVQ device IR into
fresh AOT images, costing roughly 100 seconds per link. S60 tested the supported
compiler prelink boundary once, without rebuilding the main kernel sources,
changing live archives, adding custom registration code, or touching a GPU.

The pinned image is
`sha256:e41d59c0940e1bd2f6f64c9d5d2d7452b77812b0a8407fac8d67cacec5d7fdd2`;
actual DPC++ is `2026.1.0 (2026.1.0.20260617)`. The target remains
`spir64_gen`, `bmg-g31`, per-kernel device splitting, four parallel link jobs,
and `-cl-fp32-correctly-rounded-divide-sqrt`.

Eight existing fat objects supply the common images: native MMVQ, IQ kernels,
and core device, graph, pinned, weights, layout, and memory objects. The native
object is SHA256
`64050c1ebd9e6ed31006d7c9800b0b6bf7748d23512b6c6dfecbc6c4da67e9d1`.
The frozen kernel/core archives remain respectively
`f2b53eb482814b2c9a046e524a1f76dee717fddc6fc45073a492880bb153e9a8`
and `34e91c40f01b7aa0e652e195228dfc1fadba93e40006c12788c81426567b0286`,
matching the S55 build closure. Source, headers, adapter, archives, and common
input hashes were checked before and after the build; frozen archives were
also compared with the live archives before prelinking.

## COMMAND: compiler-produced wrappers and a standard host link

[Intel's compiler reference](https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/2026-0/fsycl-link.html)
documents bare `-fsycl-link` as producing a partially linked device object for
later host linking. The [DPC++ Users Manual](https://intel.github.io/llvm/UsersManual.html#link-options)
describes the compiler wrapper's host-linkable representation. The pinned
driver accepts the bare flag; `-fsycl-link=image` explicitly fails because that
option was removed. Passing both archives directly with `-o` also failed the
driver-only probe, so the executed recipe uses the eight explicit objects.

The exact executed scripts are preserved in
[aot_prelink_s60](../strata/flash-next/lowlevel_probe/aot_prelink_s60/README.md).
They perform three link stages:

1. Bare `-fsycl-link` on the eight common objects creates
   `mmvq-core.images.o`, once for that exact source/object/compiler version.
2. Compile the S59 adapter with its reviewed flags, then separately prelink
   `iq4xs_rpw_leaf.o` into `leaf.images.o`. Its header-generated device kernels
   mean the adapter cannot simply be assumed to contain host code only.
3. Ordinary `icpx`, **without `-fsycl`**, links the adapter, both wrappers,
   frozen host/fat archives, SYCL libraries, and Level Zero. Passing `-fsycl`
   here would rediscover the original IR and undo the intended reuse.

The parent runner acquires `bin/cpu-run`; Docker has no GPU device mount and
uses `--network none`, read-only source inputs, and scratch on disk.

## RESULT: actual CPU build and final link passed

The first setup attempt stopped before `icpx`: `/usr/bin/time` was absent.
Its approximately 11 seconds and complete outputs are preserved under
`aot-reuse-s60-execution/attempt1-missing-time`. Version 2 changed only timing
to Bash's built-in `time`; no kernel changes or retries of failed compiler work
were needed. The successful CPU-run reported 101 seconds total:

| Stage | Actual seconds |
| --- | ---: |
| Common image prelink | 79.553 |
| Adapter compile | 9.665 |
| Adapter image prelink | 0.363 |
| Final ordinary host link | 0.394 |

Both build and tee exited zero. The actual verbose final link log contains one
`/usr/bin/ld` invocation and zero `ocloc`, `llvm-link`, `sycl-post-link`,
`clang-offload-wrapper`, or `llvm-foreach` actions. The map retains both wrappers'
constructor/destructor entries and device image records: eight for the adapter,
1078 for the common wrapper. The original host archive supplies native MMVQ and
IQ members. The core archive was supplied, but no host members were selected.
SYCL registration references remain dynamic library references as expected;
this is compiler-generated registration, not a custom ABI implementation.

Output SHA256 identities:

- Common wrapper: `8f5a7efea1c815b497edfaa1bd33e02d86d4cf9ceea38fd9190d44bacfd7cd21`.
- Adapter wrapper: `80356965e911ac95f83599ff8049b9a364250f9cce75d0f18f6fd9be7ed5a117`.
- S59 executable: `56e551afe6faf58d377340dd8a248674ef98e7cebb2e4f07011629ee6bfc4814`.

## VERDICT: retain this narrow reuse boundary after a bounded runtime canary

The supported CPU prelink and ordinary host link work on the pinned compiler.
Retain the common wrapper with its exact input hashes and compiler/target flags
for later adapters. Changing a common source, object, relevant header, compiler,
target option, or device dependency requires reconsidering and rebuilding that
wrapper. Other device families are not automatically covered by this component.

This preserves input IR and host ABI bindings, not identical GPU images. The
S59 72-second card0 window passes all four fresh-process numerical/full-byte,
initial/changed graph, guard, container-removal and pre/post-health gates. An
independent audit verifies eight exact directed capture chains and pointer
bindings (168 kernel nodes/160 edges). This canary covers public Q8_1 and
IQ4_XS C6 RPW1/2 only, not every common image or model/serving behavior.
It establishes neither old-AOT byte identity nor independent dispatch/ISA.
Reusing the wrapper for a second adapter should avoid the common AOT stage, but that
repeat-build saving has not been measured. There is no matched inference speed
claim and no general cache framework. S59 keeps RPW1 because its primitive
timing difference is tiny; see the separate IQ4_XS report.

Evidence root:
`/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014`.
Primary records are `aot-reuse-next-s60-prep/` (official references, exact input
inventory, help, driver-only plans), `aot-reuse-s60-execution/` (actual scripts,
first failure, `subagent-link-audit.json`, `root-runtime-canary.json`), and
`iq4xs-rpw2-s59-prep/build/`
(`common-inputs.sha256`, phase logs/times, `host-link.log`, `link.map`,
`symbols.txt`, output `sha256.txt`, and zero exit records).

Actual Opus5.5/Fable5.1 post reviews (23.972/49.426s) agree with this narrow
canary scope; root read both complete texts and excludes their unmeasured
serving/build-speed forecasts. Earlier CPU-audit pending-runtime records stay
intact and are superseded by the separate runtime-canary record.
Historical source binding is `decode-head-serving-s55/root-build-closure.json`.

Frozen S60:37files/284825bytes, with the shared S59 build/runtime evidence
referenced by its whole-manifest hash. External `aot-prelink-s60-manifest.json`
SHA256 `4ba3d0b0d0d76eace5dbfcd48e382d5b1c24918e45b4dc7ec3af9df7bc71f0e2`.
The manifest excludes itself and preserves the original setup failure.
