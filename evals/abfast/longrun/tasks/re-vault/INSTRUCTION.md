# Task: reimplement the `vault` archiver in Rust

The file `/opt/target/vault` is a stripped native x86-64 Linux executable: a command line archiver for a
custom container format ("VLT3"). Its source code is not available. Your job is to reverse engineer it and write
a clean, independent Rust reimplementation in /work/repo that behaves exactly like the original.

## What you have
- /work/repo: a Cargo project (binary name `vault`, no external crates available, build with
  `cargo build --release --offline`). `src/main.rs` is a stub. `samples/` holds a few archives made by the original
  and their source files. This statement is also saved as /work/repo/TASK.md; re-read it whenever you need.
- /opt/target/vault: run it as often as you like, in any directory. Make your own test files.
- Standard tools: objdump, gdb, strace, ltrace, readelf, nm, xxd, python3 (with capstone and pyelftools), gcc, and
  possibly others. The original runs unprivileged; ptrace based tools work.

## What to build
`target/release/vault` must reproduce the original's observable behaviour for all of its subcommands:
`digest`, `info`, `list` (with `-l`), `cat`, `extract`, `verify`, `pack` (options `-k KEY`, `-m 0|1|2|auto`).
Observable behaviour means: exit status, standard output, the bytes of files it writes (archives must be
BYTE-IDENTICAL to what the original writes for the same inputs and options, including the compressor choices,
encryption and checksums) and the exact error messages on standard error. It must also read every archive the
original can write, including encrypted ones, and handle corrupt or truncated archives the way the original does.

## Rules
- The grader builds your project in a clean container where /opt/target/vault does NOT exist, then runs a few
  hundred differential tests against recorded behaviour of the original. Anything that calls, copies, embeds or
  wraps the original binary scores zero. Write a real implementation.
- Do not use external crates; only std is available.
- You may work for several hours. Credit is partial: the grade is the fraction of feature groups
  ("milestones") that pass, and each group needs every test in it to match. The groups run from simple (checksums,
  listing) to hard (byte exact compression, error handling). Do the easy parts first, then keep going.
- Your context window is limited and will be summarized as you work: keep findings (format layout, constants,
  algorithms, open questions) in /work/repo/NOTES.md and keep your own test scripts under /work/repo/tests/.
- When you believe you are finished, test every subcommand against the original on varied inputs (empty files, large
  files, repeated data, random data, wrong keys, truncated archives) and fix every difference.
