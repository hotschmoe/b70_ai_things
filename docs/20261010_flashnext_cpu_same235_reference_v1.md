# Exact target235 independent CPU reference control

CONFIG

Use the already qualified isolated CPU V3 llama.cpp build/runtime and exact
original UD-Q4_K_XL four shards. No rebuild, quantizer/component change, GPU grant,
registry edit or captured native tensor/state/router input. The two exact T0/T1
system+user messages, rendered bytes and235 IDs are copied from the authentic
shortwarm tokenizer fixture. They are input provenance, not old response proofs.
Each target runs once in its own fresh server process, greedy temperature0,
seed1, natural64 cap, no ignoreEOS, unchanged CPU context2048/prefill64/FP16KV,
cacheOFF and no speculative decoding. No previous warm request is sent to CPU.

Exclusive pair lease excludes GPU serving, but containers have no GPU devices.
Inherited guards:112GiB start reserve,116GiB cgroup/no swap,6GiB live reserve with
immediate owned stop,8 CPU threads, exact pinned39992 compiler/runtime image,
cleanenv/emptycwd, all6ELF+resolved dependency closure, actual pre/post runtime
library checks, metadata imagec388 and same source35 tokenizer/template bytes.
Both real wrapper preflights precede full model reads; all4 publisher hashes and
both exact third-shard known pages bracket actual work and terminal teardown.

COMMAND

CPU source/control tests:
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s llamacpp/flash-next -p test_cpu_same235_reference_v1.py -q

Root-only wrapper smoke, no model mounts or inference:
python3 llamacpp/flash-next/qualify_cpu_same235_reference_v1.py --wrapper-preflight --build-root /mnt/vm_8tb/b70/build/flashnext-cpu-build-v3-20261010 --source-receipt /mnt/vm_8tb/b70/build/flashnext-cpu-source-v2-20261010/source-receipt.json --output NEW_SMOKE_DIRECTORY

Root-only exclusive-RAM actual control after GPU serving/serial147 closes:
python3 llamacpp/flash-next/qualify_cpu_same235_reference_v1.py --build-root /mnt/vm_8tb/b70/build/flashnext-cpu-build-v3-20261010 --source-receipt /mnt/vm_8tb/b70/build/flashnext-cpu-source-v2-20261010/source-receipt.json --output NEW_REFERENCE_DIRECTORY

RESULT

Source preparation only; no agent model read/inference/Docker/GPU execution.
Old CPUfunctionalV3 prompt successes are not imported as new reference answers.
Fresh metadata rendering/encoding must equal the declared original235 fixture,
and live CPU apply-template/tokenize must match the same bytes and IDs before
completion. Record complete output IDs/text/stop cause and independently decode
with the original tokenizer. EOS or the declared64 cap is a valid diagnostic
terminal, not an answer-quality PASS; even One/EOS is retained unchanged.

VERDICT

A source/lifecycle PASS means two exact-input natural CPU observations closed
with source/memory/ownership/model identity. It does not establish underlyingEOS
cause, native/CPU arithmetic equivalence, native bitwise authority, broad quality,
full48 math, cache/concurrency or speed. CPU HC activation Q8_0/Q8_K association
and GDN implementation remain distinct from native SYCL F32/Q8 paths. Root must
compare the newly observed CPU behavior to actual native exact235 observations.
