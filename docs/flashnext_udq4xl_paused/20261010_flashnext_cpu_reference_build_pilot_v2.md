# Fresh CPU llama.cpp reference build and bounded pilot, V2

CONFIG -> COMMAND -> RESULT -> VERDICT

CONFIG: Source-only proposal for the original Unsloth UD-Q4_K_XL four shards.
V1's actual snapshot receipt remains historical: its recursive exclusion of
all directories named models omitted src/models and tools/mtmd/models.
ROOT's V1 configure failed before compilation with missing models/models.h.
The retained V1 snapshot/build receipt/log in flashnext-cpu-build-v1-20261010
is a preparation failure, not model physics or CPU inference evidence.
V2 excludes ONLY the top-level models payload directory; it retains nested
architecture source. All V1 artifacts remain unchanged.

Fresh isolated CPU code is a quality comparator, not a Strata F32 arithmetic
oracle. All frozen FP64 references and failed numerical evidence remain intact.
The current source corpus is 3579 files / 98156639 bytes. The JSON source/build
plan binds each relevant source file and a full ordered path/size/mode/hash
corpus digest. ROOT's source-only producer additionally records actual HEAD,
tracked index, dirty binary diff against HEAD, untracked-source status, and a
verified copied snapshot. This agent did not invoke git or build anything.

COMMAND (ROOT, source metadata only, NEW output outside source):

```
PYTHONDONTWRITEBYTECODE=1 python3 llamacpp/flash-next/prepare_cpu_reference_source_v2.py \
  --source /mnt/vm_8tb/github/llama.cpp-flashnext \
  --plan llamacpp/flash-next/cpu-reference-source-build-plan-v2.json \
  --output /mnt/vm_8tb/b70/build/flashnext-cpu-source-v2-NEW
```

Any HEAD/corpus drift, gitlink, symlink, existing output or mid-copy mutation
fails. The snapshot excludes .git, top-level models payload fixtures, payload suffixes
at every depth, build directories,
node_modules, virtual environments and compiled artifacts; these are not inputs
to the selected build targets. Dirty/untracked source is retained, not reset. Before any git metadata or
output creation, every source_evidence and essential_sources path/hash must
exist in the manifest. The latter pins 209 architecture/CMake files, including
src/models/qwen4exp.cpp and tools/mtmd/models/models.h. The copied snapshot is
checked again. Immediately before configure, ROOT runs:

```
PYTHONDONTWRITEBYTECODE=1 python3 llamacpp/flash-next/prepare_cpu_reference_source_v2.py \
  --plan llamacpp/flash-next/cpu-reference-source-build-plan-v2.json \
  --check-snapshot <verified-snapshot-source-directory>
```

Missing architecture source, a modified expected source hash or corpus drift
fails this preconfigure gate. Tiny tests deliberately delete either required
architecture path from the manifest and require rejection.
The source receipt's full manifest, dirty.patch and metadata are authoritative;
a version string from the snapshot without .git may say unknown.

Build proposal (ROOT): use pinned image39992 from the JSON, explicit
--entrypoint /usr/bin/env, --network none, --memory 8g --memory-swap 8g --cpus 2
--pids-limit 256. Mount ONLY snapshot /source:ro and a new owned /output:rw;
no devices, GPU groups, GPU sockets, model mount or Docker socket. Create new
/output/empty-cwd and /output/tmp; working directory is the former. Pass Docker command argv starting
-i PATH=/usr/bin:/bin LANG=C LC_ALL=C OMP_NUM_THREADS=2 TMPDIR=/output/tmp
/bin/bash --noprofile --norc -c with the exact JSON configure/build argv.
GCC/G++ must exist and report GCC, not icpx/SYCL. Save executable/library paths,
SHA256, compiler/version, cmake/version, uname/lscpu, exact argv, stdout/stderr,
CMakeCache, compile_commands, container inspect, terminal exit/removal, and
source hashes before/after. Do not run install or package/dependency download. Associate the configure
stdout/stderr file SHA256 with its receipt on failure as well as success;
likewise build logs if compilation is attempted. Never discard failed logs.
Configure disables UI download/npm, LLGuidance fetch and OpenMP fetch, all GPU,
RPC, dynamic backends, repack, system GGML, BLAS and LLAMAFILE. LLAMA_BUILD_MTMD=OFF does not disable the tools mtmd subdirectory:
tools/CMakeLists.txt unconditionally adds mtmd when tools are enabled; the
root option controls only a standalone-library hook (root CMake:266-269).
Therefore V2 retains tools/mtmd/models source and does not invent an option to
remove it. Its configure/build dependency is not permission to load a vision
projector. Threads/OpenMP
are local requirements: missing dependency means FAIL, not network installation.
Parallel2 and 8 GiB are caps, not measured build memory claims.

Before any model: reject CMake unused-option warnings or enabled unexpected
backend; inspect every ELF NEEDED dependency and ldd resolution/hash for
llama-server/completion/tokenize/debug and local dependencies. Require no
SYCL, Level Zero, CUDA, HIP, Vulkan, OpenCL, RPC or external ggml dependency.
Record CPU ISA and actual library identities. Use an empty executable directory
containing only this build's binaries (no libggml-*.so plugins) and an empty cwd.
Unset GGML_BACKEND_PATH and all inherited LLAMA_ARG/GGML/GPU overrides by env -i.
Backend loader searches executable/cwd even with BACKEND_DL OFF
(ggml-backend-reg.cpp:480-506,578-604); isolation is mandatory. First run
--version/--help and --list-devices without model, recording CPU-only backend
registry. No install path is consumed; never cmake --install.

Exclusive-RAM pilot (ROOT, only after GPU serving/RAM-heavy work stops): verify
all four original shard full hashes using the existing identity tooling, fresh
known-page guards and tokenizer/template metadata. model-lock.json pins original
revision766911a6 and the exact four hashes. Mount only those four GGUF files ro;
no MTP, vision projector, native SDK, converted tensors or weights replacement.
Total shard bytes111334654784 (~103.69 GiB) versus host121 GiB leaves theoretical
17 GiB; actual peak/RSS/page-cache/swap/PLE workspace remain unmeasured. Record
MemAvailable and swap before/peak/after. Abort on memory pressure/OOM rather
than calling a partial response quality PASS. mmap/lazy-on avoids eager PLE
materialization but does not establish a memory bound.

Preferred pilot is one newly owned CPU server process per prompt, localhost,
no router, no persistent cache, then terminal teardown. Common argv:

```
-m <verified-original-shard1> --device none -ngl 0 --load-mode mmap \
--lazy-mode on --no-repack --cache-type-k f16 --cache-type-v f16 \
-c 2048 -b 64 -ub 64 -t 8 -tb 8 -np 1 --flash-attn off \
--spec-type none --no-warmup --cache-ram 0 --cache-reuse 0 \
--jinja --reasoning off --reasoning-budget 0 \
--alias hotschmoe-dd,qwen3.8-flash-next-Unsloth-UD-Q4_K_XL-llamacpp-cpu-v2 \
--host 127.0.0.1 --port <owned-unused-port>
```

No GPU enumeration is authorized just by -ngl0: CPU-only build, dependencies,
plugin isolation and backend registry must already be proven. /v1/models must
report id hotschmoe-dd and both aliases. common/arg.cpp:3024 inserts aliases into
a set; common/common.h:512 and server-context.cpp:1505-1516 choose its first
entry. Lowercase hotschmoe-dd sorts before the explicit qwen research alias;
modelpath/default names are only fallback when the set is empty. Actual API
identity still must be recorded. NO models.yaml edits: current BATCH/V9 plans
bind its SHA. This is an unregistered private diagnostic; no shelf promotion
or registry-dependent eval qualification is authorized by this proposal.

Preregistered messages, no system message, each a separate fresh process:

1. Write only a Python function add(a, b) that returns their sum. No explanation.
   Functional expectation: code parses, defines add, and add(2,3)==5,
   add(-4,1)==-3, add(0,0)==0. Optional code fences may be stripped once.
2. A box has 7 red balls and 5 blue balls. How many balls are there in total?
   Answer with only the number.
   Functional expectation: stripped response exactly 12.

For each send identical messages to /apply-template with enable_thinking=false
and save the rendered prompt. Cross-check no-thinking rendered suffix and
GGUF template SHA, do not manually invent ChatML. Save /tokenize ids with
add_special=true, parse_special=true; verify actual accepted prompt IDs match
server logs/token count and later comparison's identical IDs. Use /completion
with that exact rendered prompt, cache_prompt=false, return_tokens=true,
stream=false, temperature=0, seed=1234, n_predict=64, repeat_penalty=1,
samplers=["temperature"]. Save exact HTTP request/response, token IDs, text,
finish/cap indicators and stderr. Prompt+output must fit2048 without truncation.
Natural EOS/EOT or model template stop before64 is required. Reaching64,
reasoning despite off, malformed output or failed functional checks is FAIL.
Repeat each prompt in another fresh process to test deterministic equality;
this is four serial loads, not repeated-request state or concurrency proof.

Existing llama-debug can capture a shared-prefix final head in another fresh
exclusive-RAM process, same common geometry, --binary-file <exact-rendered>
--save-logits --logits-output-dir <new-dir> (no --verbose tensor dump).
examples/debug/debug.cpp:190-210 decodes the prompt;87-96 selects its final head;
104-157 writes native-endian float32 vocab-length .bin and int32 input IDs.
Bind complete bytes/shape/finite checks and exact IDs. It does not autoregress
or capture every generated head. It cannot consume supplied numeric IDs;
prove its tokenization equals accepted IDs, otherwise do not compare the head.
Server return_tokens records generated IDs (server-schema.cpp:34,
server-context.cpp:1972), not complete head bytes. No custom large harness needed.

RESULT: Eight CPU source/mock tests exercise source mutation/mode/symlink/stale
HEAD/output exclusion, nested architecture retention/removal negatives,
actual essential-source hashes and real flag declarations. No build, ELF/backend,
model-read, inference, speed or quality result exists from this preparation.

VERDICT: First build CPU-only, then two bounded functional prompts under
exclusive RAM. This settles whether a separate implementation produces useful
answers sooner than another broad HC observer. CPU HC Q8_0 weights use Q8_0
activation dots; Q4_K/Q5_K experts use Q8_K activations and BF16 roles use BF16
(ggml-cpu.c:272-398,1278-1328). LLAMAFILE is disabled to avoid a second fastpath.
CPU GDN rounds decayed state before the key dot (ops.cpp:11225-11252), differing
from Strata FMA association. Thus useful CPU answers/head differences diagnose
quality and source differences; they cannot erase native numerical failures,
qualify full48 arithmetic, or establish native exp/rsqrt/whole-HC equivalence.
