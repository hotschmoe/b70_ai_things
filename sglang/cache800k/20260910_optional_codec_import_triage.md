# Optional codec import triage

CONFIG -> TP1 card 1, SGLang nightly plus pointer-source repair, immutable image
sha256:1660b87ca7ab262e205132a7c23b0fd0c75aeaff91bef4ead31ddf4ad1c4f672.
COMMAND -> Inspect complete server log and extracted installed server arguments
and MiMo audio processor. Read processor-discovery and tokenizer-manager source
at the image's recorded SGLang source revision
fd70325c108c17b635e7caaa57fc7ab4e6e63ad9. Run CPU-only AST regression fixtures.
RESULT -> Missing libavutil.so.56 through .60 prevents importing the optional
MiMo audio processor, but the outer processor scanner catches Exception, logs
"Ignore import error", and continues. Application startup completed at
04:00:35; /v1/models returned 200 at 04:00:37; chat completion returned 200 at
04:02:32. Graceful shutdown began at 04:02:36 with zero requests. The later
SystemExit: 0 / CancelledError trace comes from the SIGTERM watchdog.
VERDICT -> This is not a codec-caused startup failure. Neither that optional
import warning nor the shutdown traceback proves GPU poisoning. No dependency
repair or backend rebuild is necessary for the text-only investigation.

CONFIG -> Source-level candidate text-only configuration; subsequent live test
shows it is unsupported for this Qwen architecture.
COMMAND -> Inspect --language-model-only argument and discovery predicate, then
review the parent-run warm-retry-v2 actual startup.
RESULT -> Installed argument help says this omits the multimodal encoder and
rejects multimodal requests. At the recorded source revision, tokenizer setup
skips processor discovery when get_disagg().language_model_only is true. Two
CPU/source tests verify that a simulated codec RuntimeError does not prevent the
next processor from registering, and that the text-only predicate bypasses
optional processor discovery.
VERDICT -> Do not use --language-model-only for this Qwen3_5ForConditionalGeneration
artifact on this image: warm-retry-v2 fails its architecture resolver, which
currently supports only MuseGlimmer for this mode. The CLI flag and discovery
predicate exist, but source-level acceptance did not establish model support.
Retain the ordinary multimodal setup and tolerate the unrelated MiMo codec
warning. --language-only is a different encoder/decoder-disaggregation option
and is not a substitute. No dependency repair was made by this triage.

Raw logs, extracted files, version-pinned source, and CPU tests:
/mnt/vm_8tb/b70/results/bang_isolation_20260910/sglang-codec-triage/

Source:
https://github.com/sgl-project/sglang/blob/fd70325c108c17b635e7caaa57fc7ab4e6e63ad9/python/sglang/srt/managers/multimodal_processor.py
https://github.com/sgl-project/sglang/blob/fd70325c108c17b635e7caaa57fc7ab4e6e63ad9/python/sglang/srt/managers/tokenizer_manager.py
