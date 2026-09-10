# Legacy native GDN conv-copy adapter candidate

CONFIG -> Exact7b baseline worker/model source hashes are embedded in prepare.py.
Only native271db0d4 with XPU and SD conv layout may opt in using
B70_XPU_GDN_PREFIX_CONV_COPY=1. Default0 retains the original copy functions.
The actual native publication/read probe confirmed this three-row per-prefix
contract. The two-file adapter is built in candidate image09fc6b750415; model
qualification remains pending; composed native/copy GPU contracts passed10/10
with strict health and cleanup, recorded in ../composed_boundary/20260910_result.json.

COMMAND -> prepare.py takes exact --model-source, --worker-source and fresh
--out, emits two candidate Python sources and a unified patch. test_candidate.py
runs only CPU AST-extracted source functions against the raw candidate location.
No image build, native compilation, inference or GPU execution is performed.

RESULT -> Ten TP1/TP2 source/destination address contracts pass, including
positive-bias self publication, backward publication and forward movement.
CPU copy-spec and actual shared fused helper compute the same first3-row
source bytes from block_ids[current_column + accepted_count - 1]. Exact
context initialization sees the new explicit GDN copy-function marker and
sets conv_width0, inner_size3*dim, selecting the existing generic flat-state
checkpoint-copy route for both PRE and POST. Other conv functions keep their
original rolling-window metadata. Default GDN selection, invalid flag,
non-XPU platform and wrong native identity are checked on actual source paths.

VERDICT -> Built diagnostic fallback, not a production-qualified fix. Actual native
publication/read observations reject the standalone rolling6
reference while output/SSM math passes. Retained old source reads and writes
three-row conv history per speculative column; the worker copies row offsets
inside column0. This adapter reconciles the confirmed copy contracts; its bounded composed behavior passed GPU qualification while model
qualification and same-card controls are separately pending. It does not change native arithmetic, accepted-count resets,
block allocation, scheduler policy, graph execution or other Mamba/CUDA copy
functions. Native upstream5802a414 is being evaluated independently as the
preferred producer-side repair; do not enable this legacy adapter on a rebuilt
native with a different contract. The native hash gate rejects that combination.

Before interpreting a model outcome, run the actual adapted PRE/POST byte
oracle and compose publication -> copy -> next native read against the verified
per-prefix contract, then replay the exact failing padded arrays and cached
continuations with matched settings and lifecycle evidence. Keep original
failed artifacts and do not relabel the old rolling-reference failure as a pass.
