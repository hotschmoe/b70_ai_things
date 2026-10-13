# Task: build the `rx` regular expression engine from a specification (Rust)

/work/repo contains an empty cargo project and a written specification, `SPEC.md` (about 250 lines), for a
backtracking regular expression engine with a command line protocol: pattern syntax (classes, groups, named groups,
alternation, greedy / lazy / possessive quantifiers, counted repeats, back references, lookahead / lookbehind,
atomic groups, flags `i m s x` including scoped flags), match / findall / sub commands, precise error kinds, and a
final performance requirement (exponential-time patterns must still finish quickly). Implement it as the binary
`rx` in Rust, standard library only (no external crates, no network): `cargo build --release --offline` must
produce `/work/repo/target/release/rx`. This statement is also saved as /work/repo/TASK.md.

You may NOT use any existing regex implementation (no `regex` crate; the container has none anyway) and you may not
shell out to other programs such as python or grep to do the matching.

## What you have
- `SPEC.md`: the exact semantics. Where it is silent inside the supported subset, Python 3 `re` on ASCII text is the
  reference (python3 is installed in this container, version 3.12, so you can compare behaviour while developing; the
  hidden expectations were recorded with a slightly newer Python, but only on behaviour that is identical in both).
- `tests/public_cases.json` and `tests/run_public.py`: about 30 example cases in the grader's own format with
  expected output lines (about 10 percent of the hidden suite's style). Run `python3 tests/run_public.py -v` after a build.
- `Cargo.toml` and `src/main.rs` (a stub).

## Grading
- A hidden suite of a few hundred cases in 15 milestones from easy to hard (literals, anchors, classes, groups,
  greedy quantifiers, lazy quantifiers, back references, flags, lookaround, possessive/atomic, error kinds, findall,
  sub, tricky capture semantics, performance). Each case is run as a separate process (`rx CMD FILE`) and its first
  output line is compared exactly with the expected line. A milestone passes only when all its cases pass; the grade
  is the fraction of milestones passed (the fraction of individual cases is recorded too). Partial credit is the
  point: get the basics right first, then keep adding features.
- The performance milestone allows 15 seconds per case.

## Rules
- Work in /work/repo. You may work for several hours. Your context window is limited and will be summarized as you
  work: keep design notes, decisions and known gaps in /work/repo/NOTES.md and put your own tests under /work/repo/tests/.
- When you believe you are done, write a differential fuzzer (random patterns and subjects, compare with Python's
  `re` using the same output format) and fix every difference inside the supported subset; re-read SPEC.md section
  by section, especially sections 3, 5 and 6.
