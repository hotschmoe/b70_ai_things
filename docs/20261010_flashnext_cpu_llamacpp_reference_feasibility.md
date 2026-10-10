# CPU-only llama.cpp fresh-process reference feasibility

CONFIG -> source-only audit of existing fresh checkout
`/mnt/vm_8tb/github/llama.cpp-flashnext`; campaign/initial recipe pin
`de7fa0a3c6a2e1b4cd9f22eb8d6bf5b12dbdb63b`. Existing generated build-info also
reports de7fa0a. No git/fetch command was used; exact inspected bytes are below.
COMMAND -> Read source, campaign plans, current CMake cache/build-info and host
memory metadata. No binary execution,compile,Docker,GPU,model payload or runtime
mutation; only this new ASCII document was written.
RESULT -> CPU graph/operator support exists. A fresh CPU-only build is viable,
subject to actual memory/primitive/tokenizer/fresh-process qualification.
VERDICT -> Prefer a bounded meaningful-quality CPU comparator pilot over more
unbounded HC apparatus. This is not an already trusted exact arithmetic oracle.
Frozen original FP64 references and every failed model/quality result remain.

| Tag | Checkout-relative file | SHA256 |
| --- | --- | --- |
| A | `src/models/qwen4exp.cpp` | `c30ef376bd19ea19fcd8c360ec94adb18fa6f7912911fdd4389c91c9af8606f6` |
| G | `src/llama-graph.cpp` | `06e4e4f7500ec1e87a0cde124a2568f77e4c6c3d54b4a047a2415cd4efeae7d8` |
| C | `ggml/src/ggml-cpu/ggml-cpu.c` | `45b22f8c83b2cbccabfc2f81a38ef8a5810223e1ce366b7c6f77d42faa5cc6f3` |
| O | `ggml/src/ggml-cpu/ops.cpp` | `4316ecebab08340c55a98944a30cb12241f572a507ddf9b7ee7ba24c1350cf51` |
| L | `src/llama-model-loader.cpp` | `301f8250eef5b34d14617c7c9bf5c14bb60519d199751f4d829502a5107f0593` |
| R | `ggml/src/ggml-backend-reg.cpp` | `0f63c69ad083e0744872f57d59d049b069283303f5bbc646222907d779afe4d1` |
| B | `ggml/CMakeLists.txt` | `c1d1d55416664981ea61e33b28a0dfdd36064659f15661d3a500b0de98d59ce4` |
| T | `src/llama-vocab.cpp` | `e772949cb8ad204f87845cb5e18f11e8a419fdb872e5f645012e5b6a9f84588f` |

## Model support and the arithmetic distinction

A:174-272 loads HC norm/inject F32, HC quantized down/up, linear-attention and
QSA/indexer roles, PLE table/projections/convolution and MoE. A:313-402 implements
HC mix/combine;430-500 places PLE before HC,dispatches GDN/QSA then experts/head.
A:737 onward has QSA selection;1219-1285 derives PLE rows from accepted token
history;1348-1453 gathers/dequantizes the table and applies PLE projection/history.
No new CPU model architecture port is apparent from source. O:5204-5400 supports
quantized GET_ROWS including IQ4_NL;9931 implements F32 SSM convolution;11123
implements F32 gated-delta recurrence. C supports F32/BF16/Q8_0/Q5_1/Q4_K/Q5_K/
IQ4_NL CPU traits and indexed expert multiplication. Those are source capabilities,
not complete execution or model-identity evidence.

Ordinary CPU llama.cpp is a different quantized arithmetic lane:

- A:331-335 and1376-1377 call build_lora_mm; G:1551 uses ordinary ggml_mul_mat,
  without converting HC/PLE Q8_0 weights to direct F32 arithmetic first.
- C:272-275 chooses Q8_0 activation vecdot for Q8_0 weights;1278/1328 converts
  F32 inputs to that operand type on the ordinary CPU fallback. CPU Q4_K/Q5_K
  experts use Q8_K (C:311-326), not Strata's32-element Q8_1 packet contract.
  BF16 weights likewise select BF16 CPU operands. Optimized matmul/repack routes
  need their own dispatch recording; disabling them does not remove fallback
  activation quantization.
- O:11225-11252 first rounds the decayed recurrent state,then computes the key
  dot/update; Strata forms decay*old-keydot and FMA updates. Same real recurrence,
  different F32 operation grouping. HC norm/mix and intrinsic/reduction orders
  also differ. Fresh process does not make those paths bitwise identical.

Thus CPU llama.cpp can be an independent exact-GGUF *functional quality
comparator*, after qualification. It cannot by itself adjudicate the one-ULP
HC mixed/Q8_1 midpoint mismatch or serve as an exact nativeF32 authority.
Agreement between two approximate engines also does not qualify original math.
Raw shared-prefix distributions and task correctness are useful independent
observations; preserve cross-backend differences instead of fitting thresholds.

## Fresh CPU build, isolation and memory

Current `/mnt/vm_8tb/b70/build/flashnext-llamacpp` is unsuitable as a no-GPU CPU
reference: CMakeCache has GGML_SYCL=ON, AVX2=OFF, CPU_REPACK=ON and icpx.
Existing built targets were server/bench/perplexity/primitive tests, not a newly
qualified optimized CPU-only CLI. Do not reuse that ABI tree or infer isolation
from -ngl0/--device none: linked backends may initialize before layer placement.

Use a separate immutable CPU build of the already pinned source: CPU ON, every
accelerator/RPC backend OFF, BACKEND_DL OFF, empty BACKEND_DIR, REPACK OFF,
CPU_ALL_VARIANTS OFF and BLAS/LLAMAFILE OFF initially for a simple recorded
ordinary CPU lane. Explicit AVX2/FMA/F16C settings must match1950X capability;
pin compiler/image/options and ELF dependencies. B:86/152/189-269 exposes those
choices. Build CLI,perplexity and CPU primitive/tokenizer controls; no source
patch is initially needed. Compiler/runtime libraries are freshly recorded,
not borrowed from a quarantined stack. No build command was executed here.

R:480-506/578-604 still searches backend plugin locations and GGML_BACKEND_PATH.
Therefore run from an empty isolated working directory with only the fresh CPU
binary/library set,clear backend override/plugin environment and inheritedGPU
configuration,verify no SYCL/LevelZero/OpenCL/Vulkan dependency/backend appears,
and expose no GPU devices/network in a CPU container. Backend registry must
show onlyCPU. Pure CPU inference does not need a GPU workload lease, but its
exclusive RAM reservation must prevent overlapping RAM-heavy GPU services.

111,334,654,784 model bytes =103.69GiB, not111GiB. Against~121GiB physical usable
RAM that leaves~17GiB theoretical headroom before OS/context/workspace/loader
copies. This is not admission evidence. Use original four-shard mmap loading,
lazy-mode auto/on for the large PLE tensor (L:1088-1105/1411; lazy locking is
avoided at1659), no mlock, no CPU repack/extra buffers or speculative model.
Repack includes Q4_K/Q5_K conversions despite its CMake description mentioning
Q4_0. Start one sequence,context2048 or smaller,prefill64,FP16KV; observe actual
RSS/cgroup peak/page faults/swap and failures. CPU recurrent state is roughly
113MB plus convolution state; KV/workspaces and loader transient peaks still
require measurement. Mmap address space does not require all pages resident,
but file-cache warmth and memory pressure must be retained,not changed silently.
No CPU reference may run concurrently with the RAM-heavy segmented GPU serve.

## Next concrete experiment and its limits

After current GPU work closes, reserve the host RAM window and perform only:

1. Qualify the fresh CPU build/backend isolation and locked four-shard identity;
   run CPU operator/type controls for GDN,SSM,IQ4_NL gather and mixed matmuls,
   including a wrong-input/weight negative. Preserve no-GPU enumeration evidence.
2. Cross-check exact tokenizer/template IDs against the current Strata corpus.
   T:2376-2384 provides Qwen-family BPE dispatch; that is not observed ID equality.
   Use the same already-rendered accepted IDs/text,not merely the same user phrase.
3. Run two useful frozen code/prose prompts with natural completion and declared
   cap64 for the first short pilot,one fresh process per prompt,greedy/no
   speculation/no warmup/no prompt
   reuse. Repeat each fresh process once. Retain full output IDs,finish reasons,
   task-test results and full first/shared-prefix head rows where available.
   If thinking is disabled,use the same explicitly rendered no-thinking template
   on both engines; a capped reasoning fragment is not a useful-completion PASS.
   Gate operational completion/repeatability before expanding to a frozen small
   quality/PPL corpus. Early timeouts/failures are results,not dropped samples.
4. Compare to source35 fresh single-request output under matched accepted input
   and sampling policy; separately teacher-force fixed common prefixes to avoid
   comparing distributions after generated histories diverge. Record numeric
   distances and meaningful task correctness without a newly fitted PASS bound.
   Use NEW output directories,known source pages and post-CPU newcomplete4 after
   terminal/memory pressure before returning to GPU serving.

Source/build implementation cost is low relative to a new backend: graph and
CPU kernels already exist. Actual load/inference time and peak memory are
unmeasured and could make this expensive on1950X; take a bounded first pilot
before a large evaluation. If that pilot cannot fit/finish, preserve failure and
return to the already admitted independent own-state lane; do not downgrade the
kernel or silently change model bytes to manufacture a reference.

A whole-HC arithmetic leaf remains a focused optional localization tool: native
32lane projections already match selected real up/inject cases,while norm,
ordinary sycl::exp/rsqrt,mix contraction and residual FMA still need exact raw
capture-point/intrinsic validation. That leaf can explain packet boundary
crossings; it cannot settle GDN/QSA/PLE/FFN history or useful fullmodel quality.
Do not postpone the small meaningful-prompt comparator indefinitely to perfect
a mirrored HC implementation. Neither branch changes the frozen FP64 reference,
native quantizer,prior failures,concurrency gates or latency methodology.
