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

## Independent ON admission completed

ON reader44836 subsequently finished PASS. Its readonly receipt SHA256 is
2b340fe2750bbe3067ee69a7840197935d4135dc06fbf83764b80693435544c8.
The separate admitted-history artifact binds both parent admission receipts and
rechecks all original consumed files; the preliminary artifact remains intact.

Artifact: /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/batch50-native4-offon-history-admitted-v1.json
SHA256: e2750de4c6cdddc90505edf656cec9704f69eea846b5b553acfce75432f75fce

Serial all49, complete model math, cache and latency remain unqualified.
