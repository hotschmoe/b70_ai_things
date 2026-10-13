# Task: reimplement the `vmrun` bytecode virtual machine in Go

`/opt/target/vmrun` is a stripped native x86-64 Linux executable: a runner for a custom bytecode file format
("BCX1", extension .bc). It loads a program file and executes it on a stack machine. Its source code is not
available. Reverse engineer it and write a clean, independent Go reimplementation in /work/repo that behaves
exactly like the original.

## What you have
- /work/repo: a Go module (`module vmrun`, Go 1.22, standard library only, no network). Build with
  `go build -o vmrun .`. `main.go` is a stub. `samples/` holds a few .bc programs (plain and obfuscated) together
  with the output the original printed for them (`NAME.out`). This statement is also saved as /work/repo/TASK.md.
- /opt/target/vmrun: run it as often as you like. There is no assembler: you will have to understand the file
  format and instruction set well enough to write your own test programs (for example a small Python assembler)
  and compare your runner against the original on them.
- Standard tools: objdump, gdb, strace, ltrace, readelf, nm, xxd, python3 (with capstone and pyelftools), gcc.

## What to build
`/work/repo/vmrun` must reproduce the original's observable behaviour: exit status, standard output, and the exact
text of the messages written to standard error, for every program file and command line the original accepts or
rejects:
  vmrun [-b BUDGET] [--info] PROGRAM.bc
This includes loading and validating the file (magic, version, checksum, constant pool, entry point), every
instruction (arithmetic and wraparound rules, comparisons, jumps, calls and frames, locals and globals, strings,
arrays and the heap, built-in input and output), every runtime trap (each with its own exit code and message),
the instruction budget, and the two optional encodings of program files that the header flags select. Corrupt
and malformed files must be rejected the way the original rejects them.

## Rules
- The grader builds your module in a clean container where /opt/target/vmrun does NOT exist, then runs a few
  hundred hidden test programs and compares your runner with recorded behaviour of the original. Anything that
  calls, copies, embeds or wraps the original binary scores zero. Write a real implementation.
- Only the Go standard library is available.
- You may work for several hours. Credit is partial: the grade is the fraction of feature groups ("milestones")
  that pass, and each group needs every test in it to match. The groups run from simple (header, exit codes,
  arithmetic) to hard (traps, corrupt files, encodings, limits). Do the easy parts first, then keep going.
- Your context window is limited and will be summarized as you work: keep findings (file layout, opcode table,
  encodings, trap codes, open questions) in /work/repo/NOTES.md and your own test tooling under /work/repo/tests/.
- When you believe you are finished, test every instruction and every error path against the original on varied
  programs (including malformed ones) and fix every difference.
