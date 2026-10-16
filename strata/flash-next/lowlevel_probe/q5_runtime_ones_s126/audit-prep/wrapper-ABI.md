# Actual S126 SDK raw16 wrapper ABI correction

CONFIG -> Existing V3 executable66fae0f1 and bound wrapper Info/name.
V1 HOLD is an audit assumption error, not a kernel/runtime failure.

COMMAND -> Read actual linked wrapper metadata using retained bounded S92
mmap/S107 sections helpers and exact linked Info name. No binary rehash,
archive read, decoder, compiler or GPU. One diagnostic first assertion counted
duplicate name occurrences rather than unique selected ELF bases; preserved
separately, then the existing helper's ELF-base deduplication was restored.

RESULT -> .ze_info reports two top-level by-value arguments. Arg0 is
class.sycl::_V1::range.0;8. Arg1 is class.__generated_.203;24. It is NOT two
arg_bypointer entries plus a separate4B scalar. Exact encoded user fields:

| arg index | type | payload offset | size | source offset | is_ptr |
| --- | --- | ---: | ---: | ---: | --- |
| 0 | arg_byvalue | 24 | 8 | 0 | absent |
| 1 | arg_byvalue | 32 | 5 | 0 | absent |
| 1 | arg_byvalue | 40 | 1 | 5 | absent |
| 1 | arg_byvalue | 44 | 1 | 6 | absent |
| 1 | arg_byvalue | 48 | 1 | 7 | absent |
| 1 | arg_byvalue | 56 | 8 | 8 | true |
| 1 | arg_byvalue | 64 | 8 | 16 | true |

Implicit global_id_offset0/12, enqueued_local_size12/12 and global_size72/12
are not user arg_index fields. Metadata type/offset/size/source_offset/is_ptr
are actual, not a guessed flattened capture signature.

Existing host-0.asm raw_gate at0x6480: lines16/58-61 preserve ones(r8d) via
stack0x24 to closure0x190(int32); output pointer0x198 and input0x1a0.
Lines302-308/0x6a20-0x6a4c form wrapper range0x28, copy closure's first16B
0x190->0x30, then input pointer0x1a0->0x40. Thus generated closure offsets
0..3 are ones,4..7 padding,8 outputptr,16 inputptr. The first5B metadata
slice includes the four ones bytes plus one padding byte. Pointer-valued
8B fragments use by-value+is_ptr flags. No meaning is assigned to padding.
Direct raw16 metadata separately reports int;4 arg0, int*;8 arg1,
Q81Block*;8 arg2, consistent with the original lambda's first use order.

VERDICT -> Exact wrapper-only V2 predicate corrects false shape assumption;
parser additionally retains actual source_offset/is_ptr. Projection ABI,
public/private control prefix8128, metadata/options/data/relocations, source,
artifact and host-binding gates remain unchanged. No gate relaxation,
numerical claim or new execution authorization. Stop at first continuation
failure; never auto retry or decode.
