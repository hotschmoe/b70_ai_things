# Original Flash-Next MTP intake and adapter gap, 2026-10-12

CONFIG -> Exact Unsloth UD-Q4_K_XL main artifact, pinned publisher revision
766911a6. Metadata/index/source-only inspection; no fresh model payload read,
download, GPU touch or compiler execution by the inspector.

COMMAND -> Compare existing complete GGUF inventory, current native pack index
and source52 MtpDrafter/tools against the already downloaded publisher MTP.

RESULT -> Main four shards contain1224 tensors across48 blocks and no MTP
block; current1079-entry native pack also has no MTP. A separate original
publisher artifact is already present at
models/files/qwen3.8-flash-next/unsloth-ud-q4-k-xl/MTP/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf.
It has34 tensors,2786204800 file bytes and2775258112 tensor bytes. Saved
publisher expected SHA256 is
8087dbb39fc73f79ac069b1debe4069a44b73ada900c8bd761569d43a5a90231.
The saved receipt matches; this metadata audit does NOT fresh-verify the file.
Token/vocab/merge/type/pre/EOS/BOS metadata matches the target main model.

Source52 cannot directly read it: MtpDrafter expects a dense runtime directory
and hardcoded Q2_0 expert layout rather than original blk.48 GGUF tensors.
Original gate/up experts are Q4_K, down experts Q8_0. Original draft embedding
is Q4_K and head Q6_K, whereas existing bind/record_front borrows the main
model's Q8_0 embedding/head. The current converter also changes projection
formats and applies a BF16-source norm convention. Running it on this GGUF
would not establish preservation of the original draft tensors.

Original routed expert bytes alone are1835008000 versus the current hardcoded
707788800 Q2_0 bytes. Total original draft tensor storage is about2.585GiB on
the final stage, before KV, scratch, graph and head subset buffers. Expert-cache
reserve/admission must account for this. Current mirror T1 staging explicitly
rejects MTP; do not remove that guard without separate source/runtime proof.

VERDICT -> No flag-only MTP model trial is ready. Prepare a default-off lossless
GGUF adapter using original quantized tensors, fresh-verify the separate file,
and first test a loaded draft step, target verification/rejection and state
restoration. Main-model verification stays authoritative. Useful emitted-token
rates and exact greedy OFF/ON outputs must qualify any eventual speed result.
Do not borrow another checkpoint or requantize the draft to chase a headline.

Persisted metadata audit:
/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/mixed-expert-profile-intake-v1/original-mtp-metadata-audit-v2.json
SHA256 baef5840fc93a119b2dbed86968704845f06bd4c5a3a8520f81d4a68e5feffb4.
