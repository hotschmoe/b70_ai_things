#!/usr/bin/env python3
"""Keep the Strata Flash-Next data map synchronized.

operation-data.json is the source of truth. index.html embeds a copy so it
opens from file:// without fetch/CORS. Typical maintenance:

  python3 -I build_map.py --import-ops   # refresh operations[] + rollups from S22 summaries
  python3 -I build_map.py                # re-embed operation-data.json into index.html
  python3 -I build_map.py --check        # exit 1 if embed is stale, JSON invalid, or non-ASCII

--import-ops reads only the small per-window summary.json files written by
summarize_trace.py, never the raw 300 MB unitrace JSON.
"""
import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "operation-data.json"
HTML = HERE / "index.html"
BEGIN = '<script id="map-data" type="application/json">'
END = "</script><!--/map-data-->"
S22 = "/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/draft-profile-s22"
CONTROL = S22 + "/profile/steady-summary/summary.json"
ASYNC = S22 + "/profile-async/steady-summary/summary.json"
CYCLES = 8  # eight interior replay cycles in steady-window.json

# First match wins. (stage id, device or None for any, regex on the raw operation string)
RULES = [
    ("peer_ep_wait", "GPU1", r"ep_wait_kernel"),
    ("primary_ep_wait", "GPU0", r"ep_wait_kernel"),
    ("peer_send_return", "GPU1", r"ep_send_rows"),
    ("ep_region", None, r"ep_bump_kernel|wait_flag_ge"),
    ("primary_push", "GPU0", r"ep_push_kernel"),
    ("peer_inbox_copy", "GPU1", r"ep_copy_uncached"),
    ("peer_route_plan", "GPU1", r"resident_plan_ep"),
    ("primary_route", "GPU0", r"resident_plan_ep|route_512_multi|route_96e6fb|group_resident|sigmoid_scale_rows"),
    ("peer_experts", "GPU1", r"native_gu_port|native_down_port|swiglu_q8_1_entries"),
    ("primary_experts", "GPU0", r"native_gu_port|native_down_port|swiglu_q8_1_entries"),
    ("combine", "GPU0", r"combine_k10"),
    ("target_head", "GPU0", r"WideQ5K, 6>.*\{62080;"),
    ("draft_head", "GPU0", r"WideQ5K, 1>.*\{62080;"),
    ("verify_select", "GPU0", r"sampler_greedy|mtp_select|row_top_prob"),
    ("hyperconn", "GPU0", r"gr_up_multi|gr_down_sliced|gr_norm_split|gr_down_reduce|broadcast|add_streams"),
    ("qsa_attention", "GPU0", r"qsa_decode_attn|attn_chunk|attn_merge|block_scores|block_topk|kv_append|norm_rope|window_ids"),
    ("gdn", "GPU0", r"gdn_|conv_residual|history_advance"),
    ("act_quant", "GPU0", r"quantize_q8"),
    ("draft_block", "GPU0", r"gu_grouped_t|down_grouped_t"),
    ("embed", "GPU0", r"embed_rows"),
    ("dense_proj", "GPU0", r"native_mmvq|mmvq_multi|launch_multi_n|bf16_f32_mmvf|Wide32"),
    ("host_staged_inputs", None, r"copy_from_mapped|copy_i32_from_mapped"),
    ("runtime_glue", None, r"zeCommandListAppend|copy_from_mapped|copy_i32_from_mapped|copy_indexed|copy_row_to_first|append_|apply_|post_"),
    ("host_sync_submit", "CPU", r"."),
]
GGML = {16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 42: "Q2_0"}
WIDE = {"WideQ5K": "Q5_K", "WideQ4K": "Q4_K", "WideIQ4XS": "IQ4_XS", "Wide32IQ4NL": "IQ4_NL", "Wide32Q8": "Q8_0?"}


def device(proc, kind):
    if kind == "host":
        return "CPU"
    if "0:b:0:0" in proc:
        return "GPU0"
    if "0:44:0:0" in proc:
        return "GPU1"
    return "?"


def clean(s):
    return re.sub(r"strata::kernels::|\(anonymous namespace\)::|sycl::_V1::|dpct::", "", s)


def describe(op):
    g = re.search(r"\[SIMD32 (\{[^}]*\}) (\{[^}]*\})\]$", op)
    grid, wg = (g.group(1), g.group(2)) if g else ("", "")
    body = clean(op[: g.start()] if g else op)
    if body.startswith("_ZTS"):
        m = re.findall(r"\d+([A-Za-z_]\w*?_kernel_[0-9a-f]{6})", body)
        name = m[-1] if m else body[:60]
    elif "(" not in body:
        name = body
    else:
        ids = [i for i in re.findall(r"::([A-Za-z_]\w*)", body) if i != "operator"]
        name = ids[-1] if ids else body[:60]
    t = re.search(r"(launch_\w+<[^()]*?>)\(", body)
    tmpl = t.group(1) if t else ""
    wtype = ""
    if tmpl:
        a = re.search(r"<(\d+)", tmpl)
        w = re.search(r"<(Wide\w+)", tmpl)
        if w:
            wtype = WIDE.get(w.group(1), "")
        elif a and ("port" in tmpl or "mmvq_multi_nc" in tmpl):
            wtype = GGML.get(int(a.group(1)), "ggml type " + a.group(1))
        elif "q6k" in tmpl:
            wtype = "Q6_K"
    elif "bf16_f32_mmvf" in body:
        wtype = "BF16"
    n_out = ""
    if wg == "{32; 4; 1}" and grid:
        n_out = int(grid[1:].split(";")[0]) * 4
    return name, tmpl, grid, wg, wtype, n_out


def stage_for(op, dev):
    for sid, d, rx in RULES:
        if (d is None or d == dev) and re.search(rx, op):
            return sid
    return "unattributed"


def load_ops(path):
    s = json.loads(Path(path).read_text())
    out = {}
    for o in s["operations"]:
        out[(device(o["process"], o["kind"]), o["operation"])] = o
    return s, out


def import_ops(data):
    ctl, cops = load_ops(CONTROL)
    aops = {}
    if Path(ASYNC).exists():
        _, aops = load_ops(ASYNC)
    rows = []
    for (dev, op), o in sorted(cops.items(), key=lambda kv: -kv[1]["total_ms"]):
        name, tmpl, grid, wg, wtype, n_out = describe(op)
        a = aops.get((dev, op))
        rows.append({
            "stage": stage_for(op, dev), "device": dev, "kind": o["kind"], "kernel": name, "template": tmpl,
            "grid": grid, "wg": wg, "wtype": wtype, "n_out_inferred": n_out,
            "count": o["count"], "calls_per_cycle": round(o["count"] / CYCLES, 3),
            "sum_ms_per_cycle": round(o["total_ms"] / CYCLES, 4),
            "median_us": round(o["median_us"], 3), "p95_us": round(o["p95_us"], 3),
            "min_us": round(o["min_us"], 3), "max_us": round(o["max_us"], 3),
            "async_median_us": round(a["median_us"], 3) if a else None,
            "async_sum_ms_per_cycle": round(a["total_ms"] / CYCLES, 4) if a else None,
            "raw": op,
        })
    data["operations"] = rows
    roll = {}
    for r in rows:
        k = roll.setdefault(r["stage"], {})
        d = k.setdefault(r["device"], {"ops": 0, "calls_per_cycle": 0.0, "sum_ms_per_cycle": 0.0})
        d["ops"] += 1
        d["calls_per_cycle"] = round(d["calls_per_cycle"] + r["calls_per_cycle"], 3)
        d["sum_ms_per_cycle"] = round(d["sum_ms_per_cycle"] + r["sum_ms_per_cycle"], 4)
    data["stage_rollups"] = roll
    data["operations_meta"] = {
        "control_summary": CONTROL, "async_summary": ASYNC if aops else None,
        "cycles": CYCLES, "complete_events": ctl.get("complete_events"),
        "timestamp_shift_us": ctl.get("timestamp_shift_us"),
        "gpu_busy_union_ms_8_cycles": ctl.get("gpu_busy_union_ms"),
        "note": "sum_ms_per_cycle = summed kernel service time / 8. Kernels on one card can overlap engines, "
                "ep_wait spin kernels count as busy, and the two cards run concurrently: never add these "
                "into a latency budget. Traced (instrumented) execution; clean timings live in benchmarks.",
    }
    known = {s["id"] for s in data["stages"]}
    missing = sorted({r["stage"] for r in rows} - known)
    if missing:
        print("warning: rule stages without a curated stage entry:", missing, file=sys.stderr)


def embed(data):
    text = json.dumps(data, indent=1, ensure_ascii=True).replace("</", "<\\/")
    html = HTML.read_text()
    i, j = html.index(BEGIN) + len(BEGIN), html.index(END)
    return html[:i] + "\n" + text + "\n" + html[j:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--import-ops", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    data = json.loads(DATA.read_text())
    if args.import_ops:
        import_ops(data)
        DATA.write_text(json.dumps(data, indent=1, ensure_ascii=True) + "\n")
    new = embed(data)
    if args.check:
        bad = [p.name for p in (DATA, HTML, HERE / "README.md") if not p.read_bytes().isascii()]
        if bad or new != HTML.read_text():
            print("stale embed" if not bad else "non-ASCII: " + ", ".join(bad))
            return 1
        print("ok: embed synchronized, ASCII clean, %d stages, %d operations"
              % (len(data["stages"]), len(data.get("operations", []))))
        return 0
    HTML.write_text(new)
    return 0


if __name__ == "__main__":
    sys.exit(main())
