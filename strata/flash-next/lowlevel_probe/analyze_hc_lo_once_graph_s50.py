#!/usr/bin/env python3
# ROOT ONLY after normal stop/health; reads this small probe log and 26 snapshots.
import hashlib, json, pathlib, statistics, sys
run, s48 = map(pathlib.Path, sys.argv[1:3])
rows = [json.loads(x) for x in (run / "oracle.log").read_text().splitlines() if x.startswith("{")]
footer = [x for x in rows if x.get("status") == "PASS_S50_STRICT_GRAPH_PROBE_ONLY"]
assert len(footer) == 1 and footer[0]["strict_cases"] == 1 and footer[0]["calls"] == 32
snap = [x for x in rows if x.get("type") == "snapshot"]
timing = [x for x in rows if x.get("type") == "paired_batch_graph_timing"]
assert len(snap) == 26 and len(timing) == 16
expected_phases = {"direct-a","direct-b","graph-a","graph-b","batch-a"} | {f"batch-warm-r{i}" for i in range(8)}
assert {(x["variant"],x["phase"]) for x in snap} == {(v,phase) for v in (0,1) for phase in expected_phases}
assert footer[0]["strict_snapshot_pairs"] == 5 and footer[0]["paired_warm_rounds"] == 8
assert {(x["round"], x["variant"]) for x in timing} == {(r,v) for r in range(8) for v in range(2)}
assert all(x["captured_reads"] == 32 and x["graph_submissions"] == 1 and x["warmup_batch_replays_per_variant"] == 4 for x in timing)
orders = [(0,1),(1,0),(1,0),(0,1),(0,1),(1,0),(1,0),(0,1)]
assert all(orders[x["round"]][x["position"]] == x["variant"] for x in timing)
anchors = []
for x in snap:
    assert x["T"] == 6 and x["apply"] == 1 and x["inject"] == 1
    assert x["guards"] and x["finites"] and x["tails"]
    phase = x["phase"]
    name = f'T6-a1-i1-mode{x["variant"]}-{phase}.bin'
    source = f'T6-a1-i1-mode{x["variant"]}-warm-r0.bin' if phase == "batch-a" or phase.startswith("batch-warm-r") else name
    a, b = run / "snapshots" / name, s48 / "run-clean/snapshots" / source
    # Timed final: identical 32-read recursion, same reset and fixture as S48.
    aa, bb = a.read_bytes(), b.read_bytes()
    assert len(aa) == x["bytes"] and aa == bb, name
    anchors.append({"file":name,"S48_file":source,"bytes":len(aa),"sha256":hashlib.sha256(aa).hexdigest()})
assert len(list((run / "snapshots").glob("*.bin"))) == 26
metrics = {}
for v in (0,1):
    ordered = sorted((x for x in timing if x["variant"] == v), key=lambda x:x["round"])
    metrics[v] = {}
    for key in ("enqueue_us", "after_enqueue_to_drained_us", "complete_us", "same_queue_barrier_span_us", "graph_event_span_us"):
        values = [x[key]/32 for x in ordered if x[key] is not None]
        metrics[v][key] = {"batch_mean_us_per_HC_read":values,"median":statistics.median(values) if values else None,"min":min(values) if values else None,"max":max(values) if values else None}
a = metrics[0]["complete_us"]["batch_mean_us_per_HC_read"]
b = metrics[1]["complete_us"]["batch_mean_us_per_HC_read"]
result = {"status":"NUMERICAL_ANCHORS_PASS_HEALTH_REQUIRES_ROOT_CLOSURE","anchors":anchors,"raw_timings":timing,"metrics":metrics,"paired_complete_percent_change":[100*(y/x-1) for x,y in zip(a,b)],"ratio_of_complete_medians_percent_change":100*(statistics.median(b)/statistics.median(a)-1),"max_same_variant_complete_spread_us_per_HC_read":max(max(a)-min(a),max(b)-min(b)),"limits":["Eight 32-read batch graph means per mode; no individual HC p95 or serving qualification", "Same-queue event span includes host gaps and barriers; not isolated service", "Outputs cannot identify mode; captured branch binding follows source/API contract", "Final health/identity/teardown and quiet/profiler evidence remain external"]}
# Position and time trends are retained regardless of outcome, never selected away.
result["position_and_time_trend"] = {
 str(v): {"position" : {str(pos): [x["complete_us"]/32 for x in timing if x["variant"] == v and x["position"] == pos] for pos in (0,1)},
          "first4_median_us_per_read": statistics.median((a if v == 0 else b)[:4]),
          "last4_median_us_per_read": statistics.median((a if v == 0 else b)[4:])}
 for v in (0,1)}
delta = statistics.median(b)-statistics.median(a)
result["predeclared_conservative_signal_screen"] = {
 "all8_same_sign": all(y<x for x,y in zip(a,b)) or all(y>x for x,y in zip(a,b)),
 "median_difference_exceeds_max_spread": abs(delta) > max(max(a)-min(a),max(b)-min(b)),
 "absolute_change_at_least2percent": abs(statistics.median(b)/statistics.median(a)-1)>=.02,
 "promotion_gate": False}
result["DOT"] = []
for v in (0,1):
 for kind in ("single", "batch32"):
    f = run / "snapshots" / f"mode{v}-{kind}.dot"
    data = f.read_bytes(); assert data
    result["DOT"].append({"file":f.name,"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest(),"binding_verdict":"root_inspection_needed_if_kernel_names_present; numerical outputs do not identify mode"})
print(json.dumps(result, indent=2))
