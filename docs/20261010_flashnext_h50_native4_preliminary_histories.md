# H50 native4 preliminary OFF/ON token histories

CONFIG -> Frozen H50 source37 native4 OFF and ON actual saved runs; matching
slots, input tokens, cancellation slot, max_new, SDK receipt, prepared baseline,
and card topology. OFF independent parent reader passed; ON reader remains live.

COMMAND -> Original native_histories and off_on_histories functions recollected
both saved requests.json files. All eight consumed plan/request/child/parent
files were stat-bracketed, SHA256 bound, and rechecked after comparison.

RESULT -> Three uncanceled requests have exact generated IDs and finish reasons.
Canceled request has four common generated IDs and actual cancellation terminal
in both arms. Complete cancellation counts are deliberately not an equality
requirement under the existing frozen cancellation timing contract.

Artifact: /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/batch50-native4-offon-history-preliminary-v1.json
SHA256: 43b1e6a55058753b45e1ec65e014ec69c001454be3fc406f3ba69046945345c5

VERDICT -> Preliminary saved token-history equality only. Independent ON parent
admission, three exact-prefix serial controls and all49 vector comparisons still
remain. No complete model math, cache, fairness, latency or shelf qualification.
