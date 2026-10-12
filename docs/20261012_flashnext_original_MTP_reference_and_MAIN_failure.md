# Original MTP first-step reference and MAIN diagnostic, 2026-10-12

CONFIG -> Original publisher MTP GGUF, SHA
8087dbb39fc73f79ac069b1debe4069a44b73ada900c8bd761569d43a5a90231.
Independent token248045/position0/hidden formula; original34 tensors.

COMMAND -> original_mtp53_first_step_cpu_v1.py --full-head, actual28.731s.

RESULT -> Two independently computed activation contracts are preserved.
The conditional native Q8_1 lane has identical routed IDs, indexer half
bytes and head top10; full-head NMSE6.217e-15, maxabs9.537e-7. The separate
float-activation lane changes routing and head values (NMSE0.015024),
although both lanes and native pick token12. Native captured activations
and router choices are never reference inputs. Float64 dot then F32 stores
is not a reproduction of native reduction ordering. No tolerance is fitted.

VERDICT -> First-step conditional evidence, not complete numerical/history
qualification. Full 248320 logits and 15 fields are retained under
original-mtp53-first-step-CPU-v1/report.json in the F17 evidence root; SHA
0e5cf70ef574937067e0ae0c842d30060a7f950f1e3f7baba4f468e86b811987.

CONFIG -> Real MAIN V2 diagnostic: same target code, own original draft.
COMMAND -> Default T1 self-commit attempt failed; explicit commit ENV0
retry also failed.
RESULT -> ENV0 forced rejection preserved target pick198 and main/draft
position2. Trial1 failed a combined assertion before printing values.
Posthealth, original4+MTP SHA scans, and normal owned retirement passed.
VERDICT -> Do not infer numerical drift or acceptance failure from an
ambiguous assertion. V3 adds a flushed pre-assert value row only. Sustained
64/512-token MTP equivalence, acceptance and speed remain unproven.

## Actual state-preserving MAIN control result

CONFIG -> V5 keeps original target arithmetic and draft34 weights, disables
implicit T1 commit and wraps only baseline T1 in existing checkpoint/KV
save/restore. Running state and touched pages are verified byte exact.
COMMAND -> Actual76687 PASS439s; independent root saved-evidence readback
passed, including closing source/ELF/receipt and original5 pre/post hashes.
RESULT -> Forcedreject targetpick198 equals T1; forcedaccept targetpick1
equals T1; actual draft proposal198 equals targetpick198 and is accepted.
Main/draft committed cells agree at2/4/6. All3 controls restore exactly,
normalexit0/ownedremove/posthealth/freshkernel pass.
VERDICT -> Real target integration control, not sustained64/512 equivalence
or numerical/speed qualification. The three trials use special-token
continuations and do not estimate natural draft acceptance or answer quality.
Next is separate fresh OFF/ON generation on an authentic tokenizer prompt.

F17 original-mtp53-main-v5-state-preserving-card0-run-v1/report.json SHA
8356256dcca853fa1c15c72ac68c4fd64a6119d50ae641d2fa0d746fd94944a6.
Independent original-mtp53-main-v5-public-v1.json SHA
968968a1a8f2797ebefb92108f6d5b2d633f62b721280b7b8fcc56f1cd533a99.
