# S112 pinned SDK barrier scope check - 2026-10-16

CONFIG -> Pinned image sha256:e41d59c0940e1bd2f6f64c9d5d2d7452b77812b0a8407fac8d67cacec5d7fdd2; compiler include root /opt/intel/oneapi/compiler/2026.1/include/sycl. Source-only, no compiler, GPU, kernel change, model consultation, or live source edit.

COMMAND -> docker image inspect confirmed the exact image ID. Three finite docker run Python header reads used --rm --network none --read-only --cap-drop ALL --security-opt no-new-privileges --user 1000:1000 --cpus 1 --memory 512m --entrypoint python3, no mounts and no /dev/dri. All exited 0. The first read inspected nd_item.hpp; the second inspected helpers.hpp, spirv_types.hpp, and group_algorithm.hpp; the third retained exact line excerpts below. This was light source inspection, not a build or measurement.

RESULT -> nd_item.hpp:213-220 defaults accessSpace to global_and_local. It passes getSPIRVMemorySemanticsMask(accessSpace) to __spirv_ControlBarrier(Workgroup, Workgroup, flags). The first operand is execution scope; the second is memory scope. Both are Workgroup, explicitly NOT Device. This method itself has no deprecation annotation in this header; the following mem_fence method does. SYCL 2020 removes the older nd_item barrier interface, while this SDK retains it.

helpers.hpp:189-232 defaults LocalScopeMask to WorkgroupMemory and selects SequentiallyConsistent | CrossWorkgroupMemory | WorkgroupMemory for global_and_local. spirv_types.hpp defines Workgroup scope=2, Device scope=1, and these memory-semantics bits as 0x10 | 0x200 | 0x100 = 0x310. CrossWorkgroupMemory identifies the affected global memory region; it does not widen the second ControlBarrier scope operand to Device. The complete front-end mapping is therefore ControlBarrier(exec=2, memory=2, semantics=0x310).

Exact full-header byte counts/SHA256 and verbatim, numbered excerpts are in exact-header-excerpts.json. Initial independent read records are also retained. No binary/SPIR-V/ISA decode was performed; this establishes SDK source semantics rather than emitted hardware fence instructions.

Specification comparison -> Khronos SYCL 2020 section 4.17.2 group_barrier defaults its fence scope to Group::fence_scope, the narrowest scope covering the group. For a work-group that is work_group; callers may request a wider scope. Its implicit release/acquire ordering covers global and local memory. Appendix D says nd_item::barrier was removed and replaced by group_barrier. This comparison does not substitute the modern API's default for the legacy SDK implementation: the actual legacy implementation above independently fixes Workgroup/Workgroup. SPIR-V memory-region flags and execution/memory scope operands are separate concepts.

Primary specification URLs checked 2026-10-16:
- https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html (revision 12, commit b3bcc5d5cc83217d14ebbb904e353c5a0d17f52b; sections 4.17.2, 4.15.1 and Appendix D)
- https://github.khronos.org/SYCL_Reference/iface/group-functions.html
- https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html (Memory Semantics, OpControlBarrier)

VERDICT -> The proposed counterargument that this pinned nd_item::barrier supplies DEVICE-scope fences is unsupported by, and conflicts with, its explicit source operands. This bounded inspection does not establish a universal hardware fault, observed wrong result, or a complete correctness proof of the surrounding reduction protocol. Parent owns the full release/RMW/acquire-chain assessment and any private hardened-kernel experiment. Live kernel remains unchanged; no GPU execution is authorized by this note.
