# Flash-Next paired copy-engine natural comparison, 2026-10-12

CONFIG -> Direct accepted50 plus default-off source52 transport patch; original
Unsloth UD-Q4_K_XL main tensors/math, static document expert profile unchanged.
Pair24/24, prefill512, context4096, spec2 reserved/no MTP, mirror staging1,
ESIMD0/grouped oneMKL0/short first chunk0/cacheOFF. Sole arm delta is
STRATA_COPY_ENGINE=0 versus1. Each fresh engine process evaluates the same
three authentic P2048 prompts, each repeated with fresh state, output cap512,
natural EOS enabled. CPU construction matched both commands and profile bytes.

COMMAND -> bin/gpu-run producer72698 PASS1038s; independent
validate_natural_p2048_native_v2.py reader63447 PASS. No extra full-model scan
in the independent reader. Producer's pre/post original full4, kernel window,
health, source/all8 and owned retirement evidence admitted.

RESULT -> All twelve requests emitted512 and finished length. Complete output
IDs/finish match across arms, and per-prompt fresh repeats match. The dedicated
queue was independently checked on both cards at actual native copy-only
ordinal1, with ring reuse/marker/reference release/recreation passing. This
comparison observes timings; it does not yet prove physical overlap or device
critical-path attribution.

The repeated-input warm rows avoid each process's cold request and the first
encounter with each prompt's kernel shapes:

| Case | Prefill tok/s OFF -> ON | PP elapsed reduction % | Pipe TTFT s OFF -> ON | Decode tok/s OFF -> ON |
| --- | ---: | ---: | ---: | ---: |
| prose_operations_review | 186.42 -> 194.98 | 4.39 | 11.032 -> 10.549 | 21.69 -> 21.68 |
| code_event_pipeline | 184.48 -> 194.15 | 4.98 | 11.149 -> 10.592 | 22.36 -> 22.38 |
| document_decision_memo | 182.81 -> 193.77 | 5.66 | 11.246 -> 10.616 | 20.57 -> 20.54 |

Arithmetic mean repeated-row PP elapsed reduction is about5.0%. Decode is
essentially unchanged on these rows. First warm input-first code/document
rows show larger differences, but are kept separately in the public record;
no new kernel first-use effect is pooled into the repeated-input claim.
Cold-process prefill was79.32/80.09 tok/s and cold decode15.31/18.46; those
first-use timings are not the optimization headline.

VERDICT -> Small measured prefill benefit, still far below2000input/50output
minimums. This is one ordered process-level A/B, not an interleaved ABBA or
population stability claim. No HTTP/API, full original-model numerical,
complete-cache/concurrency or shelf qualification. All length-capped outputs
remain incomplete tasks. Keep transport default-off pending broader use;
next throughput experiment is source50 pref2048 single-chunk P2048, which
isolates larger-batch efficiency/repeated streaming from copy-engine changes.

Evidence root:
/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/

- Producer natural-p2048-copy-engine-v2-run-v1/report.json SHA256
  beab962fadfa8de324462573a76726ea24643ccb70e6f915311bcc397a00263b
- Public natural-p2048-copy-engine-v2-public-v1.json SHA256
  2aaff799a6c1f319c8cdfa5560377b9e1c4a4fbed3b95b9942855f05a9bdeed4
- SDK52 receipt SHA256
  8f6c09aa60493dd6403771cd10592bdb780947110c3035379b4493d0a4298d94
- Linked pair operator report SHA256
  521432f8b4fbab1ee76e2a8d405199ef89e6430f5f138eb0d1523e2962149427
