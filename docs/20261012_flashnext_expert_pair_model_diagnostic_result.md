# Original expert pair reuse model diagnostic, 2026-10-12

CONFIG -> OriginalMTP enabled botharms, sole expert reuse0/1, same
compiled model ELF, authentic54 prompt/full64, card0. Historical completed
OFF component independently reconstructed; new ON only.
COMMAND -> Actual14886 PASS505s; new saved-data reader PASS.
RESULT -> All64 IDs/EOS and windows exact;30accept/2reject/1tail unchanged.
Actual7269 resident/17307 mirrored/54,234,726,400B and51segmentmaps,
profile/prefix hashes equal. OFFdecode16550.932ms ->ON15179.088ms,
8.2886percent less elapsed; target33window sum16355.433->14986.500ms,
median480.684->455.753ms. Newfull5/source/ELF/ownedexit0/removal/
post-health/kernel close passed. Producer232f83a04c398d1b67a263d1ae26f005021cfd364ae666380d69c4797ec8a361;
public840fdd6754d5f332e9b4faf3968fbad3414e14ce9b1fd565a45d1a56787eab6a.
VERDICT -> Positive separated-run singlecard diagnostic, not cleanABBA/
population/dualGPU speedpromotion. Actual E2 opportunities notobserved.
No smaller model, draft replacement or residency policy patch used.
Original fullgoal incl2000/50, mathematical/serving/cachefamily gates open.

| Metric | Reuse OFF | Reuse ON |
| --- | ---: | ---: |
| Decode wall ms | 16550.932 | 15179.088 |
| Target verification sum ms | 16355.433 | 14986.500 |
| Median target window ms | 480.684 | 455.753 |
| Accepted/rejected/tail windows | 30/2/1 | 30/2/1 |
| Resident experts | 7269 | 7269 |
| Mirror bytes | 54234726400 | 54234726400 |

The measurement uses the original34-tensor publisher sidecar and target
UD-Q4_K_XL on one card. It is a short diagnostic, not the two-card P2048/
GEN512 representative benchmark. The completed OFF process was reused
after repairing only its residency parser; ON is a later fresh process.
This saves redundant model execution but does not establish a clean
interleaved timing effect. Equal resolved placement and unchanged output
remove specific confounds, not all process/cache-temperature effects.

The first parent remains FAILED due to its STRP schema parser; its actual
OFF exit/EOF/health/full5/source component was independently rejoined.
No historical failed result was rewritten or promoted wholesale.
The actual number of same-expert paired entries remains unknown; no
speedup is attributed to an assumed expert-overlap population.

Evidence root: /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/
f17-source37-20261010/original-mtp53-expert-pair-model-card0-run-v3-on-only.
Public: original-mtp53-expert-pair-model-v3-public-v1.json in F17.
Reader: strata/flash-next/validate_original_mtp53_expert_pair_saved_v1.py.
