# API28 package-safe batch helper import

CONFIG -> Actual combined24 API launch from c1_api_trace.py imports serve.server
as a package. Frozen22 top-level batch_request_identity import fails before the
native engine starts. Existing prepared-v5 failure is retained, not repaired.

COMMAND -> Separate0028 Python-only patch moves the helper import after ROOT
path setup and uses serve.batch_request_identity, matching existing frontend
imports. Reconstruct all25 patches from pristinefb58; verify every51 final file.
Test actual pinned serving Python interpreter package/direct/module/trace paths
without devices or model mounts, with old package failure as negative control.

RESULT -> Actual old API container exit1/noOOM, removed, no native engine started;
pre/post per-card and compiled health and two source pages PASS. C1 qualification
FAILED149s; dependent final model hash skipped and UNOBSERVED. CPU test initially
selected systemPython and found missingJinja; preserved that test error, then use
actual /opt/b70-c1-python/bin/python as recorded launch. All4 corrected imports
PASS; old package negative reproduces missinghelper. Synthetic tokenizer settings
serve only --help and are not a model preparation or GPU qualification.

VERDICT -> Import compatibility proved only. No model math, request, cache,
quantization, native ABI or serving flags change. New immutable SDK plan requires
fresh build/source fingerprint/oracle/upload and genuine C1 qualification before
serving. Old snapshots and failedrun remain unchanged. Both script and package
resolve one canonical helper class; no wrapper-only sys.path workaround.

CPU receipt SHA256 397642ceac72400f671b3bc865145ab04fd60cf486219f9dd0ed69231943a765.
New plan SHA256 d496d4639b92e8de7241ec97122467e4e6e4428d95d47e690bf5aab258fbe028.
