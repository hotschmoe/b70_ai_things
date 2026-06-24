#!/usr/bin/env python3
"""Paired significance test for a within-architecture quant delta on one harness.

    stats.py --results results --harness bfcl --a 35b-int4 --b 35b-w8a8

Aligns the two configs' per_task pass/fail by task_id, runs McNemar's exact test on the discordant
pairs, and a paired bootstrap CI on the accuracy difference (a - b). This is how you decide whether a
5-10 pp delta is real or noise (codex's design note). Stdlib only -- no scipy.
"""
import argparse, json, math, os


def load_per_task(results_dir, config, harness):
    path = os.path.join(results_dir, config, f"{harness}.json")
    with open(path) as f:
        r = json.load(f)
    return {t["task_id"]: bool(t.get("passed")) for t in r.get("per_task", [])}


def mcnemar_exact(b, c):
    """Two-sided exact binomial p over the n=b+c discordant pairs (p=0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # P(X<=k) + P(X>=n-k) under Binom(n, 0.5); symmetric -> 2 * lower tail (capped at 1).
    cum = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * cum)


def bootstrap_ci(paired, iters=10000, seed=1234):
    """Deterministic LCG bootstrap of the mean paired diff (a-b). 95% percentile CI."""
    diffs = [int(a) - int(b) for a, b in paired]
    n = len(diffs)
    if n == 0:
        return (0.0, 0.0, 0.0)
    mean = sum(diffs) / n
    state = seed & 0xFFFFFFFF
    means = []
    for _ in range(iters):
        s = 0
        for _ in range(n):
            state = (1103515245 * state + 12345) & 0x7FFFFFFF
            s += diffs[state % n]
        means.append(s / n)
    means.sort()
    lo = means[int(0.025 * iters)]
    hi = means[int(0.975 * iters)]
    return (mean, lo, hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--harness", required=True)
    ap.add_argument("--a", required=True, help="int4 config label")
    ap.add_argument("--b", required=True, help="w8a8 config label")
    args = ap.parse_args()
    A = load_per_task(args.results, args.a, args.harness)
    B = load_per_task(args.results, args.b, args.harness)
    common = sorted(set(A) & set(B))
    if not common:
        print("no common task_ids -- cannot pair"); return
    paired = [(A[t], B[t]) for t in common]
    b = sum(1 for a, x in paired if a and not x)   # a pass, b fail
    c = sum(1 for a, x in paired if x and not a)   # b pass, a fail
    p = mcnemar_exact(b, c)
    mean, lo, hi = bootstrap_ci(paired)
    accA = sum(int(a) for a, _ in paired) / len(paired)
    accB = sum(int(x) for _, x in paired) / len(paired)
    print(f"harness={args.harness}  n_paired={len(paired)}")
    print(f"  {args.a} (int4) acc = {accA*100:.1f}%")
    print(f"  {args.b} (w8a8) acc = {accB*100:.1f}%")
    print(f"  delta (int4 - w8a8) = {(accA-accB)*100:+.1f} pp   "
          f"[95% CI {lo*100:+.1f}, {hi*100:+.1f}]")
    print(f"  discordant: int4-only-pass={b}, w8a8-only-pass={c}   McNemar exact p={p:.4f}")
    print(f"  verdict: {'SIGNIFICANT' if p < 0.05 else 'not significant'} at alpha=0.05")


if __name__ == "__main__":
    main()
