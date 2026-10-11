# Buffered model-byte recurrence on October 11

CONFIG -> Fresh source41 SDK and five-case HC/PLE upload qualification. Exact
selected UD-Q4_K_XL artifact; no model-file write or reference change. The
earlier power-down cause and the origin of this discrepancy remain unknown.

COMMAND -> Upload66245 completed in 278s with failed final model identity.
Read-only audit41744 compared the complete 49,376,141,504-byte shard3 using
ordinary buffered reads and prefaulted aligned O_DIRECT reads. It preserved
both versions of every differing page before any cache refresh.

RESULT -> All five upload processes exited0 and pre/post strict and compiled
two-card health passed. Final buffered shard3 SHA was
bcbdc43222c0bbcec162efb7969b49673742fce01d826a64643a39825e1cf4e7.
The independent direct-read SHA matched publisher
56758f40269cad5cd9b0d3d6fbae0f40f6d5be6de49e4ab392dbe83157d9cbd3.
Exactly one byte differed: offset 33,530,452,716, buffered E7 versus direct C7,
XOR 0x20. Device/inode/size/mtime/ctime stayed unchanged. The existing two
known-page sentinels also passed; they did not cover the newly differing page.

The root preserved 4 KiB page at 33,530,449,920, then used posix_fadvise DONTNEED
on that range through a read-only descriptor. The next buffered page matched
the preserved direct bytes. No file write or global cache flush occurred.
Fresh complete buffered hashes of all four shards subsequently passed in
scan 34099. Failed upload 66245 remains failed. New upload 13661 uses the fresh
identity receipt and a new output directory; its final qualification is pending.

Evidence under results/flashnext_udq4xl_20261008/f17-source37-20261010:

- source41-upload-shard3-readonly-audit-v1/receipt.json:
  aee144a467be7b8ce51e648ab960a66259ea251b8dba00705854f35b8b48980a
- source41-upload-shard3-targeted-cache-refresh-v1/receipt.json:
  a55e3813c26e9626ccc19ecd5b8dece4ea55d4467a8ac687702889018b5749b5
- source41-upload-shard3-targeted-cache-refresh-v1/fresh-full4.json:
  1d4b06612e4879682046fee9e3a698659e61aaa6c102f2e278cd4fd2087c924c

VERDICT -> Buffered model identity was restored and freshly verified. This
does not identify RAM, CPU, driver, storage or power as the cause. Unchanged
file metadata and sentinel checks cannot replace complete publisher hashes.
Do not reuse the failed upload as a successful qualification or infer model
fidelity from the restored file identity.
