S138 source-only C6 host marker identifier fix

CONFIG -> Base native TUe25708bf from S133/S134. S134 remains warm-only FAIL,
no timed requests and unknown actual GPU dispatch. S137 actual host inspection
found both logging blocks bound to OLD C2 text despite distinct guards/parser/
Info; tiny fresh icpx2026.1 -O3 precise C++20 -fsycl reproduction printed C2/C2.
Changing only C6 local static identifier printed correct C2/C6 in that reduction.

COMMAND -> External candidate.patch/inverse.patch only. Two lexical identifier
changes inside the existing C6 branch: logged -> logged_c6 in declaration and
(void) use. No live apply, build, GPU, decoder or new model consultation.
source-projection.json pins exact base/post bytes/SHA and whole-TU inverse.
Root can adapt the existing S133 three-action compile/archive/link wrapper with
the new source SHA/output stage after its current quiet collection completes.

RESULT -> Device/math helpers, captured payload/argument forwarding, C2/C6
parsers, selectors, validation, marker literal, launches and OFF path are source
byte-exact. No kernel source, CMake/header, arithmetic or flag policy change.
Only C6 block-local name is distinct; C2 logged stays unchanged. The tiny CPU
regression is already preserved, so no repeated CPU sweep is proposed.
Source projection is NOT a claim about future compiled marker/device identity.

VERDICT -> Source candidate only, awaiting ROOT/model review and actual main
build. Before any fresh C6 serving comparison, root should verify the rebuilt
C6 host marker references correct C6 text and retains distinct parser/Info,
then run both arms on SAME new binary with existing identity/output/work/health/
normal teardown gates. No claim this retroactively qualifies S133/S134 or that
S137 proved a particular compiler pass. No automatic rebuild/retry/promotion.
