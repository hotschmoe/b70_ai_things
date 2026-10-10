CONFIG -> NEW function-name adaptation successor to frozen V6/d87a.
V6 image-only smoke passed, but no V6 native compilation/runtime qualified.
The frozen proposal31d internally defines main for its included HC fixture,
then undefines main before its own int main. That undef removes the older
metadata wrapper's outer main rename, producing duplicate main definitions
and leaving rms37_frozen_proposal_main undeclared. All originals are retained.

COMMAND -> Root-only isolated compile using the unchanged frozen compile argv
with only /leaf/native_rms_rsqrt37_owned_entry_v7.cpp as the source replacement,
under the owned compile image/recipe and lease. Then after peer review use
qualify_native_rms_rsqrt37_v7.py with unchanged arguments. No agent compilation,
GPU or original model payload read is performed. V6 guarded shell is retained.

RESULT -> Source/CPU only. New tracked native_rms_rsqrt37_callable_v7.cpp is
byte-identical to the original proposal except its single int main identifier
becomes int rms37_frozen_proposal_main. The new wrapper is byte-identical to
the old metadata wrapper except its outer main macros/proposal include are
replaced by a direct include of that callable. No constructor/init workaround,
arithmetic, kernel, input, flags, library or hypothesis changes are introduced.
Source admission pins original31d and wrapper7ee authorities, validates exact
original file hashes, and checks both full-byte derivations. Any other byte
change fails. Actual and reconstructed compile recipes select the new entry.

VERDICT -> No actual new compile or GPU result. The macro regression is a
source finding; standalone root compile must confirm the new entry before
full owned qualification. All prior source and actual V5 failure stay intact.
