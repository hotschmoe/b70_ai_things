# Task: reimplement an obfuscated JavaScript command line tool in Python

`/opt/target/app.js` is a single-file, minified and obfuscated Node.js program (identifiers renamed, string table
with rotation and decoding wrappers, control flow flattening, dead code). It is a command line utility:
  node /opt/target/app.js <crc|b32|ser|de|expr|tpl|pack|unpack|seal|open> ...
Its readable source is not available. Reverse engineer it and write a clean, independent Python 3 reimplementation
`/work/repo/app.py` (standard library only) that behaves exactly like the original: for every command and
argument combination, the same standard output bytes, the same standard error text and the same exit status.

## What you have
- /work/repo: `app.py` is a stub. `samples/` has a few input files. This statement is also saved as
  /work/repo/TASK.md.
- /opt/target/app.js and `node` (v24): run the original as often as you like, on any input.
- Tools: a JavaScript beautifier is not installed, but node, python3, gdb, strace, sed/grep and so on are. Node's
  inspector (`node --inspect-brk`, `--cpu-prof`), instrumenting `require('fs')`, or simply pretty printing and
  de-obfuscating by hand or with scripts of your own are all fair game. The tool takes JSON, binary data and text
  inputs; arguments/behaviours not obvious from the usage line must be discovered by experiment.

## What to build
Run as `python3 /work/repo/app.py ARGS...` with the same working directory conventions (file arguments are read
relative to the current directory; `-` or no file argument means standard input where the original does so).
All ten subcommands must match, including their error messages and error positions, integer wraparound rules,
object key ordering quirks, template whitespace and truthiness rules, and the exact binary formats of `pack` and
`seal`. Output must be written as bytes (standard output is compared byte for byte).

## Rules
- The grader runs your program in a clean container with `node` and the original removed: shelling out to node,
  calling, embedding or wrapping `app.js` scores zero. Write a real implementation.
- Standard library only. Python 3.12.
- You may work for several hours. Credit is partial: the grade is the fraction of feature groups ("milestones")
  that pass, and each group needs every test in it to match. The groups run from simple (checksums, encodings)
  to hard (error messages, templates, binary formats). Do the easy parts first, then keep going.
- Your context window is limited and will be summarized as you work: keep findings (algorithms, constants, formats,
  quirks, open questions) in /work/repo/NOTES.md and your own test scripts under /work/repo/tests/.
- When you believe you are finished, compare your implementation to the original with generated inputs for every
  subcommand (valid, invalid, unusual) and fix every difference.
