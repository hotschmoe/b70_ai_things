# S53: verified native-shard NVMe placement (2026-10-15)

CONFIG -> The exact 47.04 GB expert shard now has a verified NVMe copy for
future campaigns. Source and destination hashes match the model lock.
SATA originals and daily configuration are preserved. No GPU test or startup
gain is claimed; both S49 arms finished using their original SATA path.

Source audit: native shard1 was on SATA /dev/sdc1; PLE shard2 already resolved
to NVMe /dev/nvme1n1p1. All 48 native expert header rows use shard1. Per-layer
EP fill serially reads three role slices and waits for H2D, then fills the
peer. Buffered pread, OS caching, assembly, transfer and per-expert waits can
all contribute; reported fill GB/s is not measured disk bandwidth. S49's
172.223 s and 82.127 s startup observations do not establish a repeatable cost.
STRATA_STAGE_TRIM is set but applies to layer-split ownership, not this EP-only
path or file-cache trimming. Runtime caching does not prove offline AOT reuse.

COMMAND -> After both S49 normal stops and post-health, one cpu-run copy to
a NEW namespace. Exclusive temporary file, source SHA while copying, fsync,
independent destination SHA readback, then a no-overwrite final hardlink.
Source inode/size/mtime are unchanged. No source payload is deleted or moved.

With P the raw copy directory below, actual command:
`B70_CPU_LOCK_TIMEOUT=0 bin/cpu-run python3 P/copy.py > P/copy.log 2>&1`.
The script requires retained plan.json; this is a one-time evidence recipe,
not a general relocation tool or automatic retry.

New native path:
`/mnt/cache/b70/qwen3.8-flash-next/ista-gsq-rco/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf`.
Original remains under `models/files/qwen3.8-flash-next/ista-gsq-rco-iq3-xxs/`.

RESULT -> 47,039,860,096 bytes copied. Copy/fsync 97.329 s; independent readback
64.519 s; total 161.851 s; process peak RSS 33,584 KiB. Both actual hashes:
`219ea929900dfa9ef091f3aa473fdba6874b65fcb36526d7d851ac9e95856d15`.
Copy time includes OS-cache/filesystem conditions, not physical SATA bandwidth.

Expected IQ3_XXS shard2 sibling links to existing identical NVMe IQ3_S PLE.
Its qualified model-lock digest is
`316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113`.
The 28.80 GB PLE file is unchanged and pinned by path/inode/size/mtime; it was
not redundantly copied or rehashed. No quant or model substitution occurs.

Future configs must set BOTH --native and --ple-gguf to new IQ3_XXS sibling
path strings. gguf_split_paths requires the sibling, and keeping an old PLE
alias string can append the same split twice, rejected by the dense loader.
Both compared arms must use identical new paths and retain served identity,
tokenizer and quant. This source audit prevents a predictable wasted startup.

VERDICT -> Content placement PASS, startup benefit unmeasured. Observe future
startup as a byproduct of the next fresh controlled serving comparison.
Do not reuse S49 SATA timings as an unchanged-path control. No new runtime
framework, compiler flag, stack change or extra GPU test for a file copy.

Raw audit: `devloop-cost-s53-prep/{findings,evidence,manifest}`. Copy P:
`/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/native-nvme-placement-s53/`.
Executed copy.py SHA256 `93f689e541d6e38179ab9522760827367192aba00fa1005a42ada44492690345`.
Frozen evidence: 8 files, 20320 bytes; manifest SHA256
`b862fd1de7372c9de9b02080188ce71b5af92d12851549f34b7ffd639649eb49`. Large model payload is outside this evidence bundle.
