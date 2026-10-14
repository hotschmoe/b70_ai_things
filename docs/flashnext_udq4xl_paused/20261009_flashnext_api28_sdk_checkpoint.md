# API28 native build and oracle checkpoint

CONFIG -> Immutable SDKplanv2 d496d463..., pristinefb58 plus25 patches including
Python-only API28 import correction. Pinned runtime/dependency/source preserved.

COMMAND -> Full fresh eight-target build under GPU lease with no devices/model
mount; compare51 source hashes and native executables to compiled24. Rebuild
full upload oracle against actual new engine receipt/libraries. Refreshall4
buffered publisher source hashes before new GPU upload.

RESULT -> SDK PASS308s, oracle compile/link PASS49s. All51 finalsource hashes
match plan. Exactly serve/server.py differs fromprior24, andALL8 native binaries
are bitwise identical. Actual package/direct/module/trace CPU import checks PASS.
Newcomplete all4 source hashes PASS. NewGPU upload has started, not yet qualified.

VERDICT -> Native build/identity and Python import compatibility only. Rebuilt
source390 GPU/lifecycle/fullposthash, actual API1/2, producer math and concurrency
remain required. Prior failed API5 preserved; no source/math/flags silently changed.

Engine receiptSHA256 7baca0cc1c652be3a859888a25889278dac9393b8af5a078db6b81c49e9013a3.
Oracle receiptSHA256 696be50fff026df96135b5b01dac66e9e4fea8c65dc778bcdbb5d359838fd206.
