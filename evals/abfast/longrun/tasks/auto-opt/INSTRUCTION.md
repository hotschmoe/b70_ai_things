# Task: make `nnjoin` as fast as possible without changing its answers

/work/repo is a Rust cargo project (`nnjoin`, std only, no external crates; build with
`cargo build --release --offline`) that solves this problem correctly but very slowly. Given N documents, each a set of
integer tokens, it prints for every document the most similar other document by Jaccard similarity (exact
fractions, ties to the lowest index). The exact input/output format and tie rules are in the comment at the top of
`src/main.rs`; the answers it prints are the specification: your faster program must print byte-identical output for
every valid input.

Your job is an optimisation campaign: profile, find the real bottlenecks, try ideas, keep what works, measure
everything. Algorithmic changes, data structure changes, parallelism and low level tuning are all fair game. You
may rewrite everything as long as the binary is still `target/release/nnjoin INPUT_FILE` and prints the same answers.

## What you have
- `src/main.rs`: the original (slow) program. Keep a copy of it as a correctness oracle.
- `tools/gen.py N SEED [VOCAB] [MINLEN] [MAXLEN]`: input generator (the grader uses the same kind of data with other
  seeds and sizes); `tools/expected.py INPUT`: slow Python oracle for small inputs; `tools/bench.sh [N] [SEED]`:
  build + generate + time. `tests/`: a few small inputs with expected outputs.
- Standard tools: perf (if the container allows it), valgrind may be missing; `time`, python3, gcc, gdb.

## How you are graded (hidden, partial credit)
The grader builds your project in a clean container with 4 CPUs (this container has 6; do not assume more than
4 cores help), generates its own inputs from seeds you cannot see, and scores milestones:
- correctness tiers on hidden inputs: small random cases, edge cases (empty documents, N=1, duplicates, heavy
  ties, repeated tokens, unusual whitespace and trailing-newline handling), medium inputs, and one large input
  with N=2500 documents (similar shape to `tools/gen.py 2500 SEED 30000 10 100`);
- a scale milestone: N=10000 short documents must finish within 20 seconds;
- speed milestones: speedup of your binary over the ORIGINAL program on the large N=2500 input, measured in the
  grader's own session as the median of 5 fresh runs: at least 1.5x, 2x, 4x, 8x, 16x, 32x, 64x, 128x. A speed
  milestone only counts when the output on that input is correct.
Every run uses a fresh process in a fresh directory with token ids randomly relabeled, so precomputed answers or
caches cannot help; the program must really solve the input it is given. Hard-coding anything about the visible
data scores nothing.

## Rules
- std only, no network, no build scripts that fetch anything. The binary must read only the file named in argv[1].
- Work scientifically: measure before and after each change, keep a log of experiments (idea, result, kept or
  dropped) in /work/repo/NOTES.md, and keep your benchmark scripts under /work/repo/tests/ or tools/.
- You may work for several hours. Your context window is limited and will be summarized as you work, so
  record what you learned (profile results, what was tried, current best time) in NOTES.md.
- Never leave the project in a state where `cargo build --release --offline` fails or the output is wrong; commit
  or copy known-good versions as you go (git is available in /work/repo).
