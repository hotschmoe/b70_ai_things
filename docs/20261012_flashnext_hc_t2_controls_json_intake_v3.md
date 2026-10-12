# HC T2 runtime V3: successful V2 controls JSON replay

CONFIG -> Actual corrected controls54636 PASSED: both stages48eligible/
48selected,3submitted/3completed; original normal close and identity remain
unchanged. Report66ef3154, source plan e72de52d, same compiled ELF41517ddc.
No compiler, native code, GPU controls or numerical change is needed.

COMMAND -> Root uses /tmp/b70-hc-t2-marked-natural-launch-v3.json: same actual
markedV2 assembly and immutable successful V2 controls, original closed V6
512-history referencebcaf3f5d. Natural MTP ON BOTH, HC reuse0/1 only. This
continues the already-authorized comparison; no rerun of controls/compiler.

RESULT -> route.counts returns int dictionary keys0/1 in memory, while the
successful producer wrote them as JSON object keys"0"/"1". FrozenV2 natural
compared the two Python objects directly and would falsely refuse before GPU.
NEWV3 normalizes parsed through json.loads(json.dumps(parsed)) BEFORE equality.
It accepts exactly its own controls or immutable V2 source-plan digest, still
requiring supplied actual controls report digest, same ELF, raw artifact/route
replay, normal ownership/full5 and complete semantic equality. It does not
weaken count values or recategorize failedV1 controls.

VERDICT -> One targeted CPU control uses ACTUAL saved54636 stdout/stderr and
report, demonstrates old direct equality fails/new JSON-schema equality
passes, and rejects altered stage0 count2. No GPU/compiler/model payload reads
or old-source edits. Current natural health/full5/source/normal EOF remains
mandatory; no clean speed/serving qualification is claimed.
