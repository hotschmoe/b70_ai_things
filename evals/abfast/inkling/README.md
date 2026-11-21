# abfast: Inkling-2-Mini IQ4_NL on llama.cpp SYCL (one card)

Status 2026-11-21: RUN and CLOSED by owner (sysadmin full, langs partial, base cancelled). See VERDICT below and JOURNAL.md (INK entries).

## VERDICT (2026-11-21)
- Inkling 2 Mini IQ4_NL lands closer to the 27B than to Flash-Next. Flash-Next stays measurably better.
- sysadmin: Inkling 20/24. 27B INT4 18/24 (Inkling +3/-1). Flash-Next IQ3_XXS 23/24 (Inkling -3: ps1-backup-script-fix, reg-file-fix, scheduled-task-xml-fix, all Windows/pwsh).
  Run: /mnt/vm_8tb/b70/evals/abfast/runs/sysI-inkling-iq4nl-llamacpp-pi160-20261121T091943Z
- langs (partial, stopped by owner after 6 tasks): Inkling 2/6. 27B 4/6 and Flash-Next 6/6 on the same tasks. All 4 Inkling fails were 2700 s timeouts on tasks Flash-Next passes.
  Run: /mnt/vm_8tb/b70/evals/abfast/runs/langsI-inkling-iq4nl-llamacpp-pi160-20261121T134850Z-STOPPED-BY-OWNER
- Harness caveat: Pi 1.6.0 here vs Pi 1.2.0 for the old arms, and parallel 1 vs 2.
- Serving on llama.cpp SYCL, card 1, -c 131072 -np 1 -kvu -ub 128 -cram 0: about 17 tok/s decode single stream. Prefill is about 213 tok/s in llama-bench pp512, and 130-180 tok/s on uncached chunks in serve. Both figures are after the local patches in
  /mnt/vm_8tb/b70/build/llamacpp-inkling/sycl-inkling-local.patch:
  - strided SCALE: 70 CPU-fallback ops per token removed.
  - get_rows block width: fixes an INT_MAX abort near 65k context.
- Two slots did not work: 2x131072 does not fit (eager-KQ compute buffer), and with -np 2 -kvu decode collapses to 0.4-2 tok/s when one slot prefills next to a decoding slot.

Pieces
- GGUF: /mnt/cache/b70/models/inkling-2-mini/Inkling-2-Mini-IQ4_NL.gguf (ggml-org/Inkling-2-Mini-GGUF, text only, no mmproj). sha256 in SHA256SUMS next to it.
- Build: /mnt/vm_8tb/b70/build/llamacpp-inkling/llama.cpp (master, contains #32694 Inkling 2 Mini and #30604 SYCL fix);
  binaries in build-sycl/bin; recipe build.sh there; commit recorded in /mnt/vm_8tb/b70/build/llamacpp-inkling/COMMIT.txt.
- Serve: evals/abfast/inkling/serve_inkling_llamacpp.sh (card 1, port 18160, alias inkling-2-mini-iq4nl-llamacpp, --jinja, -fa off,
  -c 262144 -np 2 = 2 slots x 131072 in one unified KV pool, runtime image b70/flashnext-llamacpp-runtime:neo26.22, gpu-run --card 1).
- abfast: run_arm.py --model-family inkling (default qwen = unchanged). Pi stays 1.2.0 (leave ABFAST_PI_DIR unset).
  inkling => models.json compat supportsReasoningEffort=true, thinkingFormat=openai, so Pi sends top-level
  reasoning_effort = "xhigh" (thinkingLevelMap xhigh->xhigh; off->none). llama-server turns that into the template kwarg;
  the Inkling template maps xhigh -> 0.99 ("Thinking effort level: 0.99"; "max" is also 0.99). Qwen enable_thinking is not sent.
  Caveat: reasoning_effort "none" makes llama-server drop the kwarg and the template then defaults to 0.9, so --thinking off is
  NOT a true off for Inkling. Only run --thinking xhigh.
- Template check (CPU only, already passed): python3 -I evals/abfast/inkling/check_template.py
  Tool call render = <|message_model|>NAME<|content_invoke_tool_json|>{"name":NAME,"args":{...}}<|end_message|>, tools declared as
  <|message_system|>tool_declare<|content_xml|>[...json sorted keys...]. Tool results need name or tool_call_id; arguments must be an object
  (llama.cpp parses them).

## 0. Lease and start (only after approval)
  cd /mnt/vm_8tb/github/b70_ai_things
  bin/gpu-run --status ; bin/xpu-health --card 1
  # foreground; holds the card-1 lease until stopped. Run in its own terminal / background task:
  evals/abfast/inkling/serve_inkling_llamacpp.sh 2>&1 | tee /mnt/vm_8tb/b70/results/inkling-llamacpp-serve-$(date -u +%Y%m%dT%H%M%SZ).log
  # If output is garbage (the old #29107 symptom; fix #30604 is in this build, so not expected):
  #   GGML_SYCL_ENABLE_OPT=0 evals/abfast/inkling/serve_inkling_llamacpp.sh
  # If KV or compute buffer does not fit (check the log for 'KV buffer' and 'compute buffer'): CTX=131072 first (65536/slot),
  #   then UB=256. Record whatever differs from 131072 per slot in the arm notes.
  # Stop: docker stop inkling-llamacpp ; then bin/xpu-health --card 1

## 1. Smoke
  K=$(cat /mnt/vm_8tb/b70/secrets/dd_api_key); U=http://127.0.0.1:18160
  curl -s -H "Authorization: Bearer $K" $U/v1/models | python3 -m json.tool        # id must be inkling-2-mini-iq4nl-llamacpp
  curl -s -H "Authorization: Bearer $K" $U/props | python3 -c 'import sys,json;d=json.load(sys.stdin);print(d.get("model_alias"),d.get("total_slots"),d["default_generation_settings"]["n_ctx"])'
  # authoritative template render incl. effort (compare with check_template.py output):
  curl -s -H "Authorization: Bearer $K" -H 'Content-Type: application/json' $U/apply-template -d '{"messages":[{"role":"user","content":"hi"}],"reasoning_effort":"xhigh"}' | head -c 600
  # one chat (xhigh); expect reasoning_content + content, finish_reason stop:
  curl -s -H "Authorization: Bearer $K" -H 'Content-Type: application/json' $U/v1/chat/completions -d '{"model":"inkling-2-mini-iq4nl-llamacpp","reasoning_effort":"xhigh","max_tokens":2048,"messages":[{"role":"user","content":"What is 17*23? Answer with the number only."}]}' | python3 -m json.tool | head -40
  # tool-call round trip (step 1: expect message.tool_calls[0].function.name=bash with parsed arguments; step 2: send the result back):
  curl -s -H "Authorization: Bearer $K" -H 'Content-Type: application/json' $U/v1/chat/completions -d '{"model":"inkling-2-mini-iq4nl-llamacpp","reasoning_effort":"xhigh","max_tokens":4096,"tools":[{"type":"function","function":{"name":"bash","description":"run a shell command","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}}],"messages":[{"role":"user","content":"Use the bash tool to run: echo hello"}]}' | python3 -m json.tool
  # step 2: append the assistant message from step 1 (with tool_calls and its id) and {"role":"tool","tool_call_id":"<id>","content":"hello"}, resend;
  #   expect a final text answer mentioning hello. Also confirm a 2-slot concurrent pair of the chat call both complete.
  # Pass criteria: identity ok, coherent answer (391), tool call parsed as structured tool_calls (not raw <|...|> text in content), round trip finishes.
  # Quick decode/prefill figures for the record: grep 'eval time' in the server log.

## 2. Arms (no GPU use by the client; Pi 1.2.0; parallel 2; xhigh; 32768 max output; 131072 context)
  cd /mnt/vm_8tb/github/b70_ai_things
  COMMON="--base-url http://192.168.10.5:18160/v1 --api-key-file /mnt/vm_8tb/b70/secrets/dd_api_key --model inkling-2-mini-iq4nl-llamacpp --model-family inkling --parallel 2 --thinking xhigh --max-output-tokens 32768 --context-window 131072"
  # a) sysadmin (24 tasks, cap 1800 s) first
  python3 evals/abfast/run_arm.py --tier sysadmin $COMMON --timeout-s 1800 --arm-label sysI-inkling-iq4nl-llamacpp-pi120
  # b) langs (19, cap 2700 s)
  python3 evals/abfast/run_arm.py --tier langs $COMMON --timeout-s 2700 --arm-label langsI-inkling-iq4nl-llamacpp-pi120
  # c) base (19 commit replays)
  python3 evals/abfast/run_arm.py $COMMON --arm-label baseI-inkling-iq4nl-llamacpp-pi120
  # Run dirs: /mnt/vm_8tb/b70/evals/abfast/runs/<label>-<stamp>. Check meta.json has model_family=inkling and
  # served_ids contains the alias. For each arm, also check the server log for 'tool_calls' parse errors or truncated generations.

## 3. Pair against earlier runs (report.py RUN_A RUN_B)
  RUNS=/mnt/vm_8tb/b70/evals/abfast/runs
  S=$(ls -d $RUNS/sysI-inkling-iq4nl-llamacpp-pi120-* | tail -1); L=$(ls -d $RUNS/langsI-inkling-iq4nl-llamacpp-pi120-* | tail -1); B=$(ls -d $RUNS/baseI-inkling-iq4nl-llamacpp-pi120-* | tail -1)
  python3 evals/abfast/report.py $RUNS/sysA-27b-int4-p2-pi120-20261014T040430Z $S
  python3 evals/abfast/report.py $RUNS/sysB-fn-iq3xxs-p2-pi120-20261014T083025Z $S
  python3 evals/abfast/report.py $RUNS/langsA-27b-int4-p2-pi120-20261014T001530Z $L
  python3 evals/abfast/report.py $RUNS/langsB-fn-iq3xxs-p2-pi120-20261014T045505Z $L
  python3 evals/abfast/report.py $RUNS/armA-matched-p2-20261013T140703Z $B
  python3 evals/abfast/report.py $RUNS/armB-iq3xxs-2slot-20261013T104837Z $B

## Notes for the record (CONFIG -> COMMAND -> RESULT -> VERDICT)
- Flash attention is off for Inkling (relative-position KQ bias disables it in build_attn_mha), so long-context prefill uses the eager
  KQ path; expect slower prefill than the Qwen arms and a larger compute buffer at 131072. This is a speed caveat, not a quality one.
- Wall-time and decode speed are not matched to the 2026-10-14 arms (different engine, model, no MTP); this run is a quality eval.
- Never use ports 18080/18124. Do not start hotschmoe-dd while this server holds card 1.
