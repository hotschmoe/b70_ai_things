#!/usr/bin/env python3
"""Deterministic, config-INVARIANT subset selection for the Aider polyglot benchmark.

The upstream benchmark.py does `random.shuffle(test_dnames)` with the GLOBAL random module
and NO seed, then takes `[:num_tests]`. That is NOT stable across runs/configs, so we must NOT
rely on `--num-tests`. Instead we pick the exercises ourselves, deterministically, and hand the
exact relative paths to benchmark.py via `--keywords` (which substring-matches the relative
path `<lang>/exercises/practice/<exercise>` -- a full path is unique, so no over-match).

Usage:
    select_subset.py <dataset_dir> <subset> [--seed N]
        dataset_dir : .../data/polyglot-benchmark  (has <lang>/exercises/practice/<ex>/)
        subset      : smoke | standard | full
    -> prints, one per line, the selected relative paths:
            cpp/exercises/practice/acronym
            go/exercises/practice/...

Selection is a SUPERSET-nested seeded shuffle: smoke(5) is the first 5 of standard(50) is the
first 50 of full(225) of the SAME global seeded ordering. So the same task ids pair across both
configs AND across subset levels, and a smoke task is always a member of standard/full.
"""
import os
import random
import sys

LANGS = ["cpp", "go", "java", "javascript", "python", "rust"]
SUBSET_N = {"smoke": 5, "standard": 50, "full": -1}


def all_exercise_paths(dataset_dir):
    paths = []
    for lang in LANGS:
        practice = os.path.join(dataset_dir, lang, "exercises", "practice")
        if not os.path.isdir(practice):
            continue
        for ex in os.listdir(practice):
            if os.path.isdir(os.path.join(practice, ex)):
                paths.append(f"{lang}/exercises/practice/{ex}")
    return sorted(paths)  # deterministic base order before shuffle


def select(dataset_dir, subset, seed=1234):
    paths = all_exercise_paths(dataset_dir)
    # One canonical seeded ordering; subsets are nested prefixes of it.
    rng = random.Random(seed)
    rng.shuffle(paths)
    n = SUBSET_N.get(subset, 50)
    if n is not None and n > 0:
        paths = paths[:n]
    return paths


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_dir")
    ap.add_argument("subset", choices=list(SUBSET_N))
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()
    for p in select(a.dataset_dir, a.subset, a.seed):
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
