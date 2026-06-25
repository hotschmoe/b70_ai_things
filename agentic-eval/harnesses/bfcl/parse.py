#!/usr/bin/env python3
"""parse.py -- BFCL v4 multi_turn score output -> agentic-eval parsed.json (stdout).

Usage:
    python parse.py <SCORE_DIR> [--result-dir <RESULT_DIR>]

  <SCORE_DIR>   directory holding BFCL_v4_multi_turn_*_score.json  (one file per subcategory).
                Each score file is a JSON list whose element [0] is a HEADER dict
                  {"accuracy": float, "correct_count": int, "total_count": int, ...}
                and elements [1:] are the FAILED entries only:
                  {"id": "<case id>", "valid": false,
                   "error": {"error_type": "multi_turn:...", "error_message": [...]}, ...}
                (Passing entries are counted in the header but NOT listed -- see BFCL
                 eval_runner.multi_turn_runner: only `not valid` entries are appended.)

  --result-dir  (optional) directory with BFCL_v4_multi_turn_*_result.json, used ONLY to
                recover the FULL set of attempted case ids (so per_task lists passes too).
                If absent we synthesize passing ids as "<subcat>__pass_<k>" from the header
                counts -- those are still stable for a given subset, but real ids are better,
                so the harness passes --result-dir.

Emitted schema (docs/HARNESS_CONTRACT.md):
  {"score": <overall multi_turn acc 0..1>, "score_name": "multi_turn_acc",
   "n_tasks": <int>, "per_task": [{"task_id","passed"}...],
   "extra": {per-subcategory acc, failure-type counts, ...}}
"""
import argparse
import glob
import json
import os
import sys

SUBCATS = [
    "multi_turn_base",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
    "multi_turn_long_context",
]


def _load_jsonl_or_json(path):
    """Both BFCL result files AND score files are JSON-LINES (one object per line). A score file's
    line 0 is the header dict, lines 1..n are failed entries. We also tolerate a single JSON array
    (forward-compat) by trying a whole-file parse first."""
    with open(path) as f:
        txt = f.read().strip()
    if not txt:
        return []
    # JSONL is the real format; whole-file array parse only succeeds for a 1-line file or a literal
    # array, both of which we want to accept.
    out = []
    multiline = False
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            multiline = True
            break
        out.append(obj)
    if multiline:
        # A single object spanning multiple lines, or a pretty-printed array. Parse whole file.
        obj = json.loads(txt)
        return obj if isinstance(obj, list) else [obj]
    # If the file was a single line that itself is a JSON array, unwrap it.
    if len(out) == 1 and isinstance(out[0], list):
        return out[0]
    return out


def _subcat_score_file(score_dir, subcat):
    p = os.path.join(score_dir, f"BFCL_v4_{subcat}_score.json")
    return p if os.path.exists(p) else None


def _result_ids(result_dir, subcat):
    """All attempted ids for a subcat, from the result file (JSONL of {'id':..,'result':..})."""
    if not result_dir:
        return None
    p = os.path.join(result_dir, f"BFCL_v4_{subcat}_result.json")
    if not os.path.exists(p):
        return None
    ids = []
    for row in _load_jsonl_or_json(p):
        if isinstance(row, dict) and "id" in row:
            ids.append(row["id"])
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("score_dir")
    ap.add_argument("--result-dir", default=None)
    args = ap.parse_args()

    score_dir = args.score_dir
    if not os.path.isdir(score_dir):
        # Emit a well-formed zero result rather than crash, so the campaign can record a failure.
        print(json.dumps({
            "score": 0.0, "score_name": "multi_turn_acc", "n_tasks": 0,
            "per_task": [], "extra": {"error": f"score_dir not found: {score_dir}"},
        }))
        return

    total_correct = 0
    total_count = 0
    per_task = []
    per_subcat = {}
    failure_types = {}          # error_type -> count
    failed_ids_total = 0

    # Discover which subcat score files exist (smoke may produce a subset of the four).
    present = []
    for sc in SUBCATS:
        if _subcat_score_file(score_dir, sc):
            present.append(sc)
    # Also pick up any unexpected multi_turn_* score files (forward-compat).
    for p in sorted(glob.glob(os.path.join(score_dir, "BFCL_v4_multi_turn_*_score.json"))):
        name = os.path.basename(p)[len("BFCL_v4_"):-len("_score.json")]
        if name not in present:
            present.append(name)

    for sc in present:
        path = _subcat_score_file(score_dir, sc) or os.path.join(
            score_dir, f"BFCL_v4_{sc}_score.json")
        rows = _load_jsonl_or_json(path)
        if not rows:
            continue
        header = rows[0] if isinstance(rows[0], dict) else {}
        failed_entries = rows[1:]

        correct = int(header.get("correct_count", 0))
        count = int(header.get("total_count", len(failed_entries) + correct))
        acc = header.get("accuracy")
        if acc is None and count:
            acc = correct / count
        per_subcat[sc] = {
            "accuracy": round(float(acc), 6) if acc is not None else None,
            "correct_count": correct,
            "total_count": count,
        }
        total_correct += correct
        total_count += count

        # Collect failed ids + failure-type breakdown.
        failed_ids = []
        for e in failed_entries:
            if not isinstance(e, dict):
                continue
            fid = e.get("id")
            if fid is None:
                continue
            failed_ids.append(fid)
            err = e.get("error") or {}
            et = err.get("error_type") if isinstance(err, dict) else None
            if et is None:
                et = "unknown"
            failure_types[et] = failure_types.get(et, 0) + 1
        failed_ids_total += len(failed_ids)
        failed_set = set(failed_ids)

        # Build per_task. Prefer real ids from the result file; else synthesize stable pass ids.
        attempted = _result_ids(args.result_dir, sc)
        if attempted:
            for tid in attempted:
                per_task.append({"task_id": tid, "passed": tid not in failed_set})
        else:
            # Failed ids are real; synthesize the passing ones with stable names.
            for tid in failed_ids:
                per_task.append({"task_id": tid, "passed": False})
            n_pass = max(0, count - len(failed_ids))
            for k in range(n_pass):
                per_task.append({"task_id": f"{sc}__pass_{k}", "passed": True})

    # Official BFCL multi_turn overall = UNWEIGHTED (macro) mean of the 4 subcategory accuracies
    # (eval_runner_helper.calculate_unweighted_accuracy). For a full run every subcat is 200 cases so
    # macro == micro; we report macro as the primary score to match the leaderboard, and keep the
    # micro (case-weighted) figure in extra for transparency.
    sub_accs = [s["accuracy"] for s in per_subcat.values() if s["accuracy"] is not None]
    overall_macro = (sum(sub_accs) / len(sub_accs)) if sub_accs else 0.0
    overall_micro = (total_correct / total_count) if total_count else 0.0

    out = {
        "score": round(overall_macro, 6),
        "score_name": "multi_turn_acc",
        "n_tasks": total_count,
        "per_task": per_task,
        "extra": {
            "subcategories": per_subcat,
            "overall_acc_macro": round(overall_macro, 6),
            "overall_acc_micro": round(overall_micro, 6),
            "overall_correct": total_correct,
            "overall_total": total_count,
            "failed_count": failed_ids_total,
            "failure_types": failure_types,
            # Rates over the FULL task set for the load-bearing structured-failure view.
            "force_terminated_rate": round(
                failure_types.get("multi_turn:force_terminated", 0) / total_count, 6
            ) if total_count else 0.0,
            "inference_error_rate": round(
                failure_types.get("multi_turn:inference_error", 0) / total_count, 6
            ) if total_count else 0.0,
            "wrong_func_or_state_rate": round(
                sum(failure_types.get(k, 0) for k in (
                    "multi_turn:instance_state_mismatch",
                    "multi_turn:execution_response_mismatch",
                    "multi_turn:method_invoke_order_mismatch",
                    "multi_turn:empty_turn_model_response",
                    "multi_turn:irrelevance_error:decoder_success",
                )) / total_count, 6
            ) if total_count else 0.0,
            "subcats_present": present,
        },
    }
    json.dump(out, sys.stdout)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
