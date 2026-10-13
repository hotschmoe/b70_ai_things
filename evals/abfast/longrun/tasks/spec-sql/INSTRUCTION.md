# Task: build the `minisql` SQL engine from a specification (Python)

/work/repo contains an empty project and a written specification, `SPEC.md` (about 300 lines), for a small SQL
engine: CREATE/INSERT/SELECT/UPDATE/DELETE/DROP, expressions with three-valued logic, scalar functions, aggregates,
GROUP BY/HAVING, joins, subqueries (also correlated), UNION, transactions, a write-ahead log file format, and a
command line front end. Implement it as `/work/repo/minisql.py` in pure Python 3.12, standard library only. You may
NOT use the `sqlite3` module (or any other SQL engine) in your implementation: it must be your own parser and
executor, and the grader reads the source. This statement is also
saved as /work/repo/TASK.md.

## What you have
- `SPEC.md`: the exact semantics. Where it is silent inside the supported subset, SQLite 3.46 behaviour is the reference.
  The `sqlite3` module and `sqlite3` command line tool exist in this container so that you can compare behaviour
  yourself while developing (as an oracle for your own tests, not inside `minisql.py`).
- `tests/public_cases.json` and `tests/run_public.py`: about 30 example cases in the grader's own format (a sample of
  the hidden suite's style, with expected results). Run `python3 tests/run_public.py -v` from /work/repo.
- `minisql.py`: a stub with the API names.

## Grading
- A hidden suite of a few hundred cases, grouped into 15 milestones from easy to hard (expressions, tables and
  basic queries, WHERE, ORDER/LIMIT/DISTINCT, functions, aggregates, GROUP BY, UPDATE/DELETE, constraints and error
  codes, transactions, joins, subqueries and UNION, correlated subqueries, the log file format, the command line).
  Each case is a script of statements with expected rows / row counts / error codes, run against a fresh database in
  its own Python process (30 s limit). A milestone passes only when every case in it passes; the grade is the
  fraction of milestones passed (the fraction of individual cases is recorded too). Partial credit is the point: do
  the easy groups first, then keep going through the harder ones.
- Row order is only compared when the query has ORDER BY; unordered results are compared as multisets. Floats are
  compared with a tiny tolerance but int and float are different types.
- The log file is also checked independently: the grader parses the file your `Database` wrote and replays it with
  SQLite, and opens files produced by the reference implementation (including torn tails) with your code.

## Rules
- Work in /work/repo. The result must be importable as `import minisql` with the working directory containing
  `minisql.py`. Helper modules are fine if they live next to it, but a single file is simplest.
- You may work for several hours. Your context window is limited and will be summarized as you work: keep design
  notes, decisions and the list of known gaps in /work/repo/NOTES.md, and put your own tests under /work/repo/tests/.
- When you believe you are done, write many extra queries of your own, run each through both your engine and the
  `sqlite3` module, and fix every difference inside the supported subset; re-read SPEC.md section by section.
