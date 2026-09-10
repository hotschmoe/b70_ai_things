# Portable seeded FP32 Gumbel assessment

CONFIG -> SGLang 2f7393f0d245bfaa8ebe0b9d0533432e6b3bd9ed, held R276
PyTorch 2.13.0+xpu on CPU. No sampler activation, model, image change, or GPU
execution. The source has no existing equivalent seeded FP32 route: ordinary
seeded sampling uses FP64; the FP32 exponential helper is unseeded.

COMMAND -> Run probe.py in the existing R276 image without device/network
access, bounded to 2 CPUs/4 GiB, with candidate and source mounted read-only:

```text
python3 /candidate/probe.py --device cpu --source /source
```

RESULT -> 100,000 random hashes plus adversarial endpoint/midpoint hashes are
finite and repeatable. Maximum FP32 error against the ideal midpoint FP64
inverse-CDF is 9.104801890202907e-7. A 30,000-trial binary categorical check gives
P(token1)=0.7988 against expected 0.8. Forbidden (-inf) token scores remain
well-defined at both endpoint hashes. Source/reference arithmetic review found
no blocker. Raw evidence:
`/mnt/vm_8tb/b70/results/bang_isolation_20260910/sglang-seeded-gumbel/cpu.json`.

VERDICT -> The proposed stable FP32 arithmetic is a viable candidate for a
small leased XPU probe. Distribution and repeatability checks support further
testing; they do not establish full sampler qualification or same-seed token
identity. Nothing is installed or enabled.

## Why the complement matters

For lower-half hashes, use u=(h+0.5)/2^32 and -log(-log(u)). For upper-half
hashes, first compute the exact INTEGER complement d=(2^32-1)-h, then
v=(d+0.5)/2^32 and -log(-log1p(-v)). Never convert a near-maximum hash to FP32
before complementing: it would round to 2^32 and lose the small tail.

The implementation reflects both branches into (0,0.5]. All intermediates stay
finite, including operations on the unselected torch.where branch. It requires
int64 conversion/comparison/subtraction and ordinary FP32 log/log1p; no FP64
operation runs on the device in the proposed path. The current Triton Murmur
hash stays unchanged and uses uint64 seed splitting and uint32 wrapping math.

## This changes the legacy noise mapping

Current SGLang uses h/(2^32-1) plus FP64 endpoint clamps. The probe executes the
actual sampler function body on CPU, substituting precomputed hashes and
exposing pre-argmax scores; its torch.compile decorator is disabled for this
arithmetic inspection. It confirms the copied legacy arithmetic exactly.

Midpoints intentionally differ. For h=0, legacy noise is -709.7827 while the
midpoint result is -3.129995. Near the maximum hash, half-step placement changes
the tail too. The large maximum difference from the legacy reference is an
expected mapping change, not FP32 numerical instability. Both are discrete
approximations driven by 32-bit hashes; a finite empirical test cannot prove
perfect categorical sampling for every vocabulary/logit configuration.

If this route is adopted, ordinary and speculative seeded sampling must share
it. Updating only verification would fail strict same-seed ordinary-sequence
parity. Even with a shared route, FP32 rounding can change close argmax ties.
No cross-backend seed-token identity is claimed.

## Position and request semantics

Ordinary seeded sampling hashes the INPUT token position, not the generated
token's position. For prompt length 7, prefill uses position6 to sample the first
generated token. The next verify tree has root input position7; a linear chain
then uses positions8,9,... . Use actual verify-input positions for each row and
repeat the request seed by verify width. Do not repeat one position over the
whole verify segment or blindly add one to these positions.

The scoped future target-only port should retain a single draft chain, with
ordinary filtering and identical masks at each conditional prefix. Dynamic
penalties and custom processors still need separate handling. TP ranks must
share target decisions. Cross-request batching must not alter seed/position
keys. Upstream only constructs sampling_seed tensors in deterministic mode;
the original request's seed must also be checked before deciding support.

## Prepared XPU probe

After parent-managed stack health and a single-card lease/device pin, run the
same script in the held Torch/Triton image with `--device xpu`. It loads the
pinned existing murmur_hash.py directly, compares 5x257 actual Triton hashes
against the Python integer reference, and runs FP32 noise/endpoint/distribution
checks. It does not import a model or serving stack. The probe rejects changed
murmur_hash.py or sampler.py source bytes and records both reviewed hashes.

```text
python3 /candidate/probe.py --device xpu --source /source
```

No unleased GPU command is provided. CPU mode labels its hash computation as a
Python reference; actual XPU integer/hash/log1p viability remains untested.
