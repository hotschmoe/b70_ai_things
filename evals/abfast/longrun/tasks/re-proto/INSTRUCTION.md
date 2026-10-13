# Task: reimplement the `kvd` network server in Python

`/opt/target/kvd` is a stripped native x86-64 Linux TCP server: a small in-memory key-value service with a custom
binary protocol (framing, a challenge-response login, typed values, a logical clock for expiry, transactions and
change notifications). Its source code is not available. Reverse engineer it and write a clean, independent,
wire-compatible reimplementation as a single Python 3 script (standard library only) at /work/repo/kvd.py.

## What you have
- /opt/target/kvd: run it with `/opt/target/kvd --port 5000 --seed 7` (listens on 127.0.0.1 only; pick any free port;
  `--seed` controls the login nonces). Talk to it with your own client scripts; open several connections.
- /work/repo/samples/transcripts.txt: hex dumps of a few recorded sessions (client '>' and server '<' bytes per
  connection) from the original, with their seeds. /work/repo/kvd.py is a stub. This statement is saved as
  /work/repo/TASK.md.
- Standard tools: objdump, gdb, strace, ltrace, readelf, nm, xxd, python3 (with capstone and pyelftools), gcc, nc,
  and possibly others.

## What to build
`python3 /work/repo/kvd.py --port N --seed S` must behave byte-for-byte like `/opt/target/kvd --port N --seed S` on
every connection: the greeting, every reply (including the request-id echo), the pushed notifications, error
frames and exactly when connections get closed, for all operations the original understands (including the ones
that never appear in the sample transcripts: find them in the binary), for valid and malformed input, for partial
frames and pipelined frames, with several clients connected at once. Everything is deterministic: there is no wall
clock in the protocol, time only moves when a client asks for it.

## Rules
- The grader starts your server in a clean container where /opt/target/kvd does NOT exist (network: loopback only),
  replays a few hundred recorded multi-connection sessions and compares every byte the original produced. Wrapping,
  proxying, embedding or executing the original binary scores zero. Python standard library only; single process
  (select/selectors/asyncio/threads are all fine as long as per-connection ordering is preserved).
- Credit is partial: the grade is the fraction of feature groups ("milestones") that pass, and each group needs
  every session in it to match. The groups run from simple (greeting, framing, ping) to hard (login hash,
  transactions, notification ordering, malformed input). Start with the easy ones, then keep going.
- You may work for several hours. Your context window is limited and will be summarized as you work: keep findings
  (frame layout, opcode table, constants, algorithms, error codes, open questions) in /work/repo/NOTES.md and keep
  your own test scripts under /work/repo/tests/.
- When you believe you are finished, compare your server against the original with scripted sessions covering every
  operation, error path and role, with several connections, and fix every difference.
