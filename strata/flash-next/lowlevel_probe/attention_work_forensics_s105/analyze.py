from pathlib import Path
import hashlib, json, re
R = Path("/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014")
S = R / "attn-value-resident-s94-serving"
OUT = Path(__file__).parent
SRC = Path("/mnt/vm_8tb/b70/build/strata-ab/src-ep")
def pin(p):
    b=p.read_bytes()
    return {"path":str(p),"bytes":len(b),"sha256":hashlib.sha256(b).hexdigest()}
pins=[]
arms={}
for mode in ("mode0","mode1"):
    paths=[S/mode/"engine-final.log", S/mode/"prefill/p4096-0.json", S/mode/"config.json"]
    pins.extend(pin(p) for p in paths)
    log=paths[0].read_text()
    start=log.index("strata trace: request 4059 0")
    segment=log[start:]
    end=re.search(r"^strata serve: prompt 4059.*\n(?:strata serve: suffix drafts:.*\n)?",segment,re.M)
    assert end is not None
    segment=segment[:end.end()]
    summary=re.search(r"^strata serve: prompt 4059.*$",segment,re.M).group()
    w=[{"position":int(a),"T":int(b)} for a,b in re.findall(r"^strata trace: window (\d+) (\d+)$",segment,re.M)]
    api=json.loads(paths[1].read_text())
    for i,x in enumerate(w[:-1]):
        x["accepted_from_next_position"]=w[i+1]["position"]-x["position"]-1
        assert 0 <= x["accepted_from_next_position"] < x["T"]
    accepted_known=sum(x["accepted_from_next_position"] for x in w[:-1])
    offered=sum(x["T"]-1 for x in w)
    assert offered == api["timings"]["draft_n"]
    suffix=re.search(r"suffix drafts: (\d+) windows, (\d+) of (\d+) drafts accepted",segment)
    suffix=dict(zip(("windows","accepted","offered"),map(int,suffix.groups())))
    arms[mode]={"windows":w,"count":len(w),"sum_T_minus_1":offered,"T_histogram":{str(t):sum(x["T"]==t for x in w) for t in sorted({x["T"] for x in w})},"accepted_first_119_from_positions":accepted_known,"accepted_final_inferred_from_total":api["timings"]["draft_n_accepted"]-accepted_known,"suffix":suffix,"non_suffix_totals_by_subtraction":{"offered":offered-suffix["offered"],"accepted":api["timings"]["draft_n_accepted"]-suffix["accepted"]},"api_usage":api["usage"],"api_work":{k:api["timings"][k] for k in ("cache_n","prompt_n","predicted_n","draft_n","draft_n_accepted")},"output_sha256":api["output_sha256"],"native_summary":summary,"native_suffix_summary":suffix,"prompt_evidence":[line for line in segment.splitlines() if any(v in line for v in ("prompt start","prefill quantum:","read 4052 tokens","read 6 tokens","prompt done","captured the","value pipeline selected"))]}
    (OUT/(mode+"-request-native-excerpt.txt")).write_text(segment)
a,b=arms["mode0"],arms["mode1"]
assert len(a["windows"])==len(b["windows"])==120
diff=[{"index_zero_based":i,"ordinal_one_based":i+1,"OFF":x,"ON":y} for i,(x,y) in enumerate(zip(a["windows"],b["windows"])) if x!=y]
# Accepted-from-next-position differs at an earlier index only when it is genuinely a different commit.
geometry_diff=[{"index_zero_based":i,"ordinal_one_based":i+1,"OFF":x,"ON":y} for i,(x,y) in enumerate(zip(a["windows"],b["windows"])) if (x["position"],x["T"])!=(y["position"],y["T"])]
source=SRC/"sycl/src/program/generate.cpp"
expected=json.loads((R/"attn-value-resident-s94-prep/source-pins.json").read_text())["before"]["sycl/src/program/generate.cpp"]["sha256"]
assert pin(source)["sha256"]==expected
source_ranges={"sycl/src/program/generate.cpp":[(650,665),(710,725),(1875,1883),(1908,1916),(2351,2366),(8934,8945),(11048,11058),(11772,11815),(11845,11861),(11871,11879),(11930,11946)],"src/spec/draft_policy.cpp":[(10,24),(38,105)],"include/strata/spec/draft_policy.hpp":[(1,17),(22,39)]}
excerpt=[]
for name,ranges in source_ranges.items():
    p=SRC/name;pins.append(pin(p));lines=p.read_text().splitlines()
    excerpt.append("FILE "+str(p)+" SHA256 "+pin(p)["sha256"])
    for lo,hi in ranges:
        excerpt.extend(str(i)+": "+lines[i-1] for i in range(lo,hi+1))
(OUT/"source-excerpts.txt").write_text("\n".join(excerpt)+"\n")
facts={"status":"RETAINED_LOG_FORENSICS_NOT_CAUSAL_RESOLUTION","arms":arms,"geometry_differences":geometry_diff,"all_window_record_differences":diff,"earliest_geometry_difference":geometry_diff[0],"rejoin_after_shift":{"index_zero_based":32,"OFF":a["windows"][32],"ON":b["windows"][32]},"S94_pinned_generate_source_exact":True,"draft_policy_source_binding":"Current source only; no retained S94 policy-object or source pin established by this task","input_pins":pins}
(OUT/"findings.json").write_text(json.dumps(facts,indent=2)+"\n")
print(json.dumps({"status":facts["status"],"first_difference":geometry_diff[0],"geometry_difference_count":len(geometry_diff),"window_counts":[a["count"],b["count"]],"offered":[a["sum_T_minus_1"],b["sum_T_minus_1"]]}))
