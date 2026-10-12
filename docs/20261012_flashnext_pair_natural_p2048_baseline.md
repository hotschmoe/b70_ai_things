# Flash-Next paired natural P2048 baseline, 2026-10-12

CONFIG -> Original Unsloth UD-Q4_K_XL, accepted SDK50, two B70s split
24/24 layers, prefill512, context4096, spec2 reserved pack capacity, no MTP,
mirror T1 staging1, ESIMD0, grouped oneMKL0, short first chunk0, cache OFF.
Three tokenizer/template-derived 2048-token prose/code/document prompts, each
repeated with fresh conversation state, in one engine process. EOS IDs248044
and248046 honored; output cap512, no forced output or padding.

COMMAND -> run_natural_p2048_native_v1.py with the selected source50 pair
configuration through bin/gpu-run. Producer69849 PASS622s; independent
validate_natural_p2048_native_v1.py reader86920 PASS. CPU-only tokenizer
output decoding25331 PASS10.3s, no model payload/GPU touch.

RESULT -> All six emitted512 non-EOS tokens and finished length. Complete
output IDs repeat exactly per prompt. Source/all8 executable identities,
original four full shard hashes, pre/post health, kernel window and owned
teardown passed. Only the first request is process-cold; an input-first
request in the same process is not process-cold. Input repeats do not reuse
prefix state because cache is disabled.

| Case | Process/input | Prefill tok/s | Decode tok/s | Pipe TTFT s | Token gap P50/P95 ms |
| --- | --- | ---: | ---: | ---: | ---: |
| prose_operations_review | cold/first | 98.62 | 19.44 | 23.292 | 45.78 / 53.35 |
| prose_operations_review | warm/repeat | 185.17 | 21.68 | 11.103 | 45.58 / 53.06 |
| code_event_pipeline | warm/first | 183.15 | 22.26 | 11.226 | 42.37 / 57.44 |
| code_event_pipeline | warm/repeat | 183.93 | 22.39 | 11.178 | 42.06 / 57.13 |
| document_decision_memo | warm/first | 183.07 | 20.40 | 11.234 | 48.14 / 58.71 |
| document_decision_memo | warm/repeat | 183.06 | 20.54 | 11.234 | 47.96 / 58.42 |

Prefill and decode rates above derive from actual native DONE counts/times.
TTFT/gaps are root native-pipe observations, not OpenAI HTTP/client metrics.
Observed prefill progress512/1024/1536/2047 corroborates the requested chunk
size; the final input token is handled by the target path. No auto/fallback
chunk selection is claimed. There is one cold process and five warm requests;
this is not a population-level TTFT percentile estimate.

Decoded first outputs are task-related partial operational analysis, event
pipeline code and a decision memo. All hit the cap: code ends mid-assignment,
and the memo includes an unsupported date. Non-EOS length eligibility does
not prove factual quality, runnable code, or natural task completion.

VERDICT -> Warm prefill183-185 tok/s and decode20.4-22.4 tok/s remain below
2000/50 targets. This is a sustained representative baseline, not a matched
optimization win or verified shelf. Independent original full-model fidelity,
complete-state cache, real concurrent streams, physical traffic and correlated
device critical path remain open. Next measured candidates are dedicated
copy-engine prefill overlap and mixed-profile expert residency, separately.

Evidence root:
`/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/`

- Producer natural-p2048-native-v1-run-v1/report.json SHA256
  4e738312ec0e00d0712b78300ce305103f1540f8111f24692612bd44876b89f2
- Public natural-p2048-native-v1-public-v1.json SHA256
  00261157dee03d2de9beac250442f505690d3d11bd757da335cb0a58840aced4
- Decoder natural-p2048-native-v1-decoded/decoded.json SHA256
  15d010bbf61fd6d4b2eddb0e67a3e61a68941dbe1e2da6fa0cb62b9afcfaeced
- Corpus natural-p2048-tokenizer-v1/corpus.json SHA256
  61b5caabf76bfedfda334973741366a8341e9f600cf00b42ac042f6700afdbef
