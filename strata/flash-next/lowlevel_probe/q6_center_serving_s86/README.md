# S86 source-only Q6 serving proposal

Exact small copies from q6-center-serving-s86-prep-v2 under the shared result
root. Original source81558e95 remains unchanged. git apply --check PASS against
that current external source; no patch applied, build, GPU or model call.
Only PF0/C6/RPW1/A2/SG32 and the two S85 dimensions are eligible. Flag
STRATA_Q6_CENTER is default OFF and process-latched. Other dispatches retain
original code. V2 excludes unsupported positive-PF fallthrough too.

Future root work: refresh upstream, actual source review, audited main build
including restored S79 host files, then matched same-binary two-card EP OFF/ON.
Current generic03bb artifacts do not represent restored source. S85 primitive
results do not establish serving gain. record-copies.json binds original files.
