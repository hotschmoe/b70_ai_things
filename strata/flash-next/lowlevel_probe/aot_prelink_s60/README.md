# S60 supported AOT prelink reproduction

These two files are byte-for-byte copies of the successful 2026-10-15 CPU-only
execution scripts. They preserve the actual absolute experiment paths and
compiler image. They are a recorded run recipe, not a general build/cache tool.

See [the dated findings](../../../../docs/20261015_strata_aot_prelink_reuse.md)
for inputs, measured build phases, identities, limitations, and primary docs.

`build-v2.sh` acquires the CPU lease and runs the pinned container without GPU
devices or network. `build-container-v2.sh` prelinks the eight existing common
objects, compiles/prelinks the S59 adapter, then performs a standard host link
without `-fsycl`. It verifies the frozen input manifest before and after.

The preserved parent invokes the container script from the original external
`aot-reuse-s60-execution` directory. Both scripts expect the S59 frozen adapter,
archives, and manifest under `iq4xs-rpw2-s59-prep`, and a prepared writable
`build/tmp` and `build/cache` on disk. The original successful output already
has `build/build.exit`, so the guard intentionally prevents rerunning over it.
For a future adapter, deliberately prepare a new output directory and matching
input manifest; preserve the recorded run. Root owns execution and GPU gates.

Exact script SHA256:

- `build-container-v2.sh`: `820e741b698096bfc55fa95e6cfb49aaa21d89edc9bf310ebf4cea60cd1e1fee`.
- `build-v2.sh`: `b7170f02ac1c3b0740efef365383e15829cd13e56bc1805dc471447c5f7de7c9`.
