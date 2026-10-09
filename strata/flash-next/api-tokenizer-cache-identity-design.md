# Exact-artifact API, tokenizer and disk-cache identity

CONFIG -> Strata fb58e0dbc8399662c0e47c76578c6e878b14f6cf with current source
patches; selected Unsloth UD-Q4_K_XL revision766911a. CPU only. No model
inference, GPU discovery, external checkout changes, JOURNAL edits or commits.
Python3.12.13, regex2026.9.10, Jinja3.1.6 and MarkupSafe3.0.3 in an isolated
/tmp environment. Source/tokenizer/dependency hashes are in the receipt.

COMMAND -> Audit server.py, strata_tokenizer.py, iq_pack.py, actual selected
GGUF metadata and the consumed session identity implementation. Prepare007;
apply the current series in a private overlay; export the actual GGUF tokenizer;
serve the real Python API over localhost with the real StrataEngine transport
connected to a protocol-only child; record received GEN token IDs. Compile
actual CPU session-file implementation under ASan/UBSan with fatal sanitizer
errors. Reproduce:
/tmp/strata-api-identity-venv-20261009/bin/python strata/flash-next/test_api_identity_cpu.py.

RESULT -> Patch0007 SHA256:
161bd77aa52aca33205f7879559bbc72d7e6ee863ae6cc3db03c3485344228b4.
Actual /v1/models lists hotschmoe-dd FIRST and the detailed research alias
qwen3.8-flash-next-Unsloth-UD-Q4_K_XL-Strata-SYCL-nativeHC-v1 second, with
alias_of and matching identity metadata. Two21-token prompts (ordinary and
Unicode) arrive unchanged as actual GEN IDs at the protocol-only subprocess;
IDs match the tokenizer built directly from the pinned GGUF and the same
rendered original template. Unknown explicit served IDs reject before GEN.
Seven identity negatives pass. Actual CPU disk-session write/read succeeds
under its original identity and rejects the changed native/artifact identity.
Host ASan/UBSan pass after the empty-hash-update correction described below.
Receipt: api-identity-cpu-receipt.json.

VERDICT -> Actual API code, tokenizer export provenance, consumed GEN transport
IDs and CPU disk identity refusal are qualified within this fixture's scope.
The subprocess performs NO inference and its log's artificial token rates are
NOT model performance evidence. No real C++ model consumption, generated
coherence, GPU state reuse, cross-backend tokenizer oracle parity, prefix-state
fidelity, concurrency or healthy GPU serving is proven by these checks.

## Existing source findings

- server.py already orders the primary served name before aliases. Its default
  primary was qwen3.8-flash-next; unknown requested model IDs silently used the
  primary. Names alone carried no artifact/tokenizer/template receipts.
- iq_pack.py exports tokenizer files only when vocab/template are absent. A
  stale existing export can therefore remain. It may also copy an existing
  base tokenizer. Loading a pack is insufficient tokenizer provenance evidence.
- strata_tokenizer.extract writes vocabulary IDs, token types and the original
  GGUF template, but originally reserialized its merge-rank dictionary.007
  preserves the exact original merge metadata order. The selected artifact's
  runtime ranks still equal its direct-GGUF ranks.
- server.py ignored tokenizer.json pre/special-ID settings and substituted a
  generic source template when the exported template was missing. Strict
  identity mode now rejects that fallback and uses actual pre/special IDs.
- SYCL's generate.cpp labeled disk state as CUDA and omitted native-HC/custom
  artifact identity from its config fingerprint. The source model fingerprint
  samples model inputs; it is not a fresh complete-shard SHA256 verification.

## Optional strict manifest source contract

Config key artifact_identity_manifest enables007; unset retains the existing
API path. Strict mode binds this selected source repo/revision/artifact and
its exact tokenizer GGUF SHA256. Primary model_name must be hotschmoe-dd;
missing model_name receives that primary, while an explicitly different one
fails. One detailed research alias is required and must agree with config.
Only known explicit request IDs are accepted; an omitted model can use the
primary. Both listed IDs describe the same immutable loaded artifact.

The manifest has these fields:

- schema:1, primary_model_name, research_alias
- model:{repo,revision,artifact,tokenizer_gguf,tokenizer_gguf_sha256}
- tokenizer_files: SHA256 for vocab.json, merges.txt, token_type.json,
  tokenizer.json and chat_template.jinja
- runtime:{exe_sha256,args,env,python_versions,python_sources}

runtime.args is the final Python server engine argument vector, excluding its
implicit --serve. Include effort_end arguments if explicitly configured.
runtime.env contains all effective STRATA_, SYCL_ and ONEAPI_ variables except
STRATA_ARTIFACT_IDENTITY_SHA256, which is injected from manifest bytes. It must
reflect the actual launch environment rather than a guessed subset. Source and
package contracts come from serve.artifact_identity.source_contract() and
python_contract(); they bind the API/renderer/tokenizer implementations and
Python/regex/Jinja/MarkupSafe versions. The new module validates original
metadata arrays/template against the actual GGUF, beyond file-hash agreement.

Runtime guard validates executable bytes, final args, effective environment,
Python sources/dependencies and tokenizer/template files before starting or
restarting the child. The actual executable and files are not substituted to
make validation pass. /v1/models adds compact artifact, source revision,
quantization scheme, tokenizer-GGUF and template identity metadata. No local
filesystem paths or implementation details are added to ordinary chat output.

Production manifest creation belongs after intake and the final binary build.
The CPU fixture's manifest binds a PROTOCOL STUB and must NEVER be deployed as
a model serving configuration. The full model's weight identity still needs
the intake/full-shard receipts and live model qualification; checking the
first metadata GGUF does not freshly hash every large weight shard.

## Prefix identity and invalidation boundary

007 identifies the consumed SYCL backend as sycl. Its disk identity includes
sycl-state-contract-v7, the artifact manifest SHA256 and explicit native HC/
legacy HC/gr_v3/qfuse/prefill HC/write-mix flags plus SYCL compile options.
Strict manifest SHA256 additionally binds the complete effective math/runtime
environment, executable, args, tokenizer/template and Python implementation.
A changed contract gives a different config fingerprint. The actual CPU file
reader rejects that old identity before restoring its payload. Native HC disk
save/restore without an artifact digest fails closed; ordinary CLI memory
prefix reuse remains available within one process.

In-memory live/parked/slot prefix state belongs to an engine process whose
loaded math and artifacts are fixed. It does not currently attach this manifest
to every checkpoint. A new engine process starts without the previous process's
GPU state; external disk restoration uses the fingerprint above. Changing
math globals or artifacts inside a running process is unsupported and is NOT
made safe by007. Real tests must still show old identity refusal, fresh-process
isolation and all state components restored correctly, including GDN, QSA,
indexer and PLE. Exact matching token IDs are necessary, not sufficient.

## Host identity bug discovered by this fixture

The first CPU fixture exposed UBSan in SessionHasher::update(nullptr,0): when
pending bytes existed, it passed a null source to zero-length memcpy and then
formed null-pointer arithmetic. That source run is unqualified.007 adds an
immediate zero-length return, preserving the digest semantics. The corrected
fixture sets -fno-sanitize-recover=all and passes actual session-file roundtrip
and changed-identity refusal without sanitizer output. This shared CPU file
has no SYCL shadow override and is the consumed implementation.

## Remaining gates

Compile007's generate.cpp and shared conversation_file.cpp in the independent
full engine overlay, then integrate a production manifest with the intake
builder after the verified binary exists. Query the actual GPU server's
/v1/models and compare its consumed IDs with these fixtures and an independent
trusted tokenizer reference. Cross-check detailed IDs in evaluation config
before scoring. Verify actual cached/new token counts, cached/uncached logits,
state isolation, eviction/cancellation and concurrent prefix use. No verified
shelf entry or model speed/stability claim follows from this CPU/API fixture.
