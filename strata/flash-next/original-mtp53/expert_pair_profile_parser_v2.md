# Actual SDK52 profile parser correction

CONFIG -> Executed matched-pair V1 produced64 original-MTP OFF tokens and normal
native teardown/health/full5, then its metadata parser refused the profile.
ON did not execute. Original source/model/kernel/compile artifacts remain intact.

COMMAND -> Runner V2 uses the actual SDK52 version1 STRP schema:24-byte header,
n_ranked packed uint16 layer/expert pairs, then the writer's int32 inverse rank
table. The profile prefix fingerprint covers exactly4 bytes per consumed pair.
The old compiled model assembly is explicitly pinned as a4abbe0c..., while V2
owns only corrected parser/consumer source. No fake relabel or model recompile.

RESULT -> Actual saved profile is196632 bytes:98304 ranked-pair bytes and98304
inverse-table bytes after the24-byte header. SDK expert_cache.cpp118-128 reads
uint16 pairs; tools/make_profile.py68 writes <HH and line70 writes inverse ranks.
The saved OFF replay passes all64 token/window/phase and actual residency parses:
7269 resident,17307 mirrored,54234726400 mirror bytes,51 segments,592 phase rows.
The replay is saved-capture metadata only, with no new runtime/model admission.

VERDICT -> V1 remains a failed experiment. Runner V2 requires unchanged actual
compile/object/operator pins and all original live lifecycle/full5 gates. It can
reuse the already-built native ELF. Source and metadata checks do not establish
ON output equivalence or a new speed result.
