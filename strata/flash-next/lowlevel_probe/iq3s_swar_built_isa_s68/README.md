# S68 recorded built SWAR ISA audit

Exact copies of the executed extraction, decoder and summarizer helpers for
the [dated findings](../../../../docs/20261015_strata_iq3s_swar_built_isa.md).
They retain actual experiment paths and expect to live in the external raw
stage directory, with S65/S66 siblings. These are reproducibility records,
not a general extraction framework or commands to overwrite frozen outputs.

Original stage:
`/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/iq3s-swar-built-isa-s68-prep/`.

`extract.py` selects the two short private C6 kernel names, saves their existing
zebins/metadata and verifies the retained public image's canonical S66 hash.
It reads existing executable metadata and three small images, never model
payload. Exact kernel-section selection excludes the auxiliary symbol-table
program. The preserved initial selector failure is in the raw stage.

`decode.sh` was invoked under `bin/cpu-run`; it runs TWO offline ocloc decodes
in the pinned image without DRI/network. Both private text names fit ordinary
filenames. No alias, new compilation or runtime/link input change is performed.
`summarize.py` records compiled resources, code bytes, exact static instruction
evidence, the canonical public-image match and source/build pins. Missing
spill/scratch fields are not proof of zero runtime usage; static counts do not
establish dispatch, latency or a serving improvement.

Prepare a NEW external output directory and explicit input paths for a later
audit. Preserve the recorded stage. Do not execute the repository copies in
place or rerun against frozen evidence. This documentation update ran no new
tests, builds, decoder commands or GPU work.

Recorded helper SHA256:

- extract.py: `23db207698ed2a83ef74781b3133d645c5bfde849f314496000479bd91898a03`.
- decode.sh: `53a944f66dc22750eaf804bb8f78dc0bfe273eb66b008fb2fb45362317f263b0`.
- summarize.py: `5fe58f30863d28b2473eba05dff65c145efae14ce9627f2fe295af7847d60eda`.
