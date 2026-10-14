# HC two-row complete operator result

CONFIG -> Fresh HC projection/composition and kernel-only fixture from exact
SDK52; cardinality2 on card0, tile flag0 so reference stays scalar, same native
F32/Q8 lane arithmetic. Synthetic original-shaped weights; no model mount.
The candidate borrows only the first20480 floats of a51200-float parts bank.

COMMAND -> Root bin/gpu-run --card0 (actual spelling --card 0), actual20716
PASS38s. run_hc_t2_operator_v1.py consumes actual compiler receipt
c13468d7a888765aeb51ba7b8b10cf946f7f43b53f6fdbb47a122f031e37e84c and
ELF78bf32ae3389592461ec38ae772c548486ed8c1aa55fc74b6996edc93f11f8d4.
Root independently checked exact native marker, typed normal terminal,
foregroundEOF/absence, health-result raw hashes, and absence of model mounts.

RESULT -> PASS exactbytes for R, xn, lo, gate, rs, mixed and injection output,
with pendingwrite0/1, direct execution and graph initial execution/replays.
The borrowed-bank tail stays byte-exact. Original queue drains and graph/USM
retirement finish normally; pre/post health and fresh kernel window pass.
External producer at
/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/hc-t2-full-operator-card0-run-v1/report.json
SHA256cc5b51f7c5f7487b3e63a14552bd30c0052aa81e1c377e17d74e6a1b01d3ff00.
Independent sibling hc-t2-full-operator-card0-public-v1.json
SHA25615f3da81890278d03b97df734b6a4629140c3ef5f0177d41b9869a641a32d8e9
binds all saved artifact hashes.

VERDICT -> Full composition operator prerequisite passes; not merely a
projection-only numerical check. This does not establish original model
fidelity, stage-local caller execution, two-GPU target equivalence or speed.
The paired caller V2 changes only eligibility guards and still needs fresh
compilation and actual-model qualification. No timing benefit is claimed.
