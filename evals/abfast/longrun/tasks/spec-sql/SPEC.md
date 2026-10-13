# minisql specification (version 1)

You implement `minisql.py`: a small in-process SQL engine in pure Python 3.12 (standard library only; the
`sqlite3` module must NOT be used by your implementation). The grader compares your engine with SQLite 3.46 on
the subset below, so whenever this document is silent about a detail inside the supported subset, SQLite's
behaviour is the reference. Anything marked "unspecified" is never tested.

## 1. Files and API

`/work/repo/minisql.py` must provide:

```python
class SQLError(Exception):          # every error the engine reports
    code: str                       # one of the error codes in section 8
    msg: str                        # free text, never tested

class Result:
    rows: list[tuple]               # SELECT: result rows; other statements: []
    columns: list[str]              # SELECT: output column names; other statements: []
    rowcount: int                   # INSERT/UPDATE/DELETE: affected rows; SELECT: -1; DDL/BEGIN/COMMIT/ROLLBACK: 0

class Database:
    def __init__(self, path=None)   # None or ":memory:" -> memory only; otherwise a log file (section 10)
    def execute(self, sql, params=()) -> Result
    def close(self)                 # flush/close the log file; an open transaction is discarded
```

* `execute` runs exactly ONE statement; one trailing `;` and surrounding white space are allowed. Anything
  after it is a syntax error.
* `?` placeholders are numbered left to right and bound from `params` (a sequence of None/int/float/str). A wrong
  number of parameters is a `syntax` error.
* Values are Python `None` (NULL), `int`, `float`, `str`. Booleans never occur; truth values are the integers 1 and 0.
* A statement that fails changes nothing (statement atomicity), even if it fails halfway through a multi-row INSERT
  or UPDATE. The database stays usable after an error. `SQLError` is the only exception type tests expect.
* Also provide `python3 minisql.py DBFILE` (section 11).

## 2. Lexical rules

* SQL keywords, function names, table and column names are case-insensitive (ASCII). Names are stored as written
  (the first CREATE spelling is used for output column names of `SELECT *`).
* Identifiers: `[A-Za-z_][A-Za-z0-9_]*`, or double-quoted `"any text"` (no escapes inside; may contain spaces or
  reserved words). Quoted names are still case-insensitive.
* Reserved words (not usable as unquoted identifiers): SELECT FROM WHERE GROUP BY HAVING ORDER LIMIT OFFSET DISTINCT
  AS AND OR NOT IN IS NULL LIKE BETWEEN EXISTS CASE WHEN THEN ELSE END JOIN INNER LEFT OUTER ON UNION ALL ASC DESC
  CREATE TABLE IF INSERT INTO VALUES UPDATE SET DELETE DROP BEGIN COMMIT ROLLBACK PRIMARY DEFAULT UNIQUE.
  (`key`, `integer`, `text`, `real`, `count`, ... are NOT reserved.)
* String literals: `'...'` with `''` for a quote. Numeric literals: `123`, `1.5`, `.5`, `3.`, `1e3`, `2.5E-2`;
  integers without `.`/exponent are `int`, others `float`. A leading minus is the unary operator.
* White space (including newlines and tabs) separates tokens. There are no comments.
* Any other character, or an unterminated string/identifier, is a `syntax` error.

## 3. Values, comparison, affinity

* Storage classes: NULL < numbers (int and float compare numerically, `1 = 1.0` is true) < text (compared by
  Unicode code point). A number is always less than any text, no conversion happens in comparisons.
* Column affinity applies when a value is stored by INSERT or UPDATE (never in comparisons):
  * `INTEGER` column: a float with an integral value becomes int (`5.0` -> `5`); a text that is entirely an
    optionally signed decimal integer becomes int (`'7'` -> `7`). Other values are stored unchanged.
  * `REAL` column: an int becomes float; a text that is entirely a number (int or float syntax, optional sign,
    optional surrounding spaces) becomes float. Other values unchanged.
  * `TEXT` column: an int becomes its decimal text. Floats into TEXT columns are unspecified.
  * NULL stays NULL.
* Truth: NULL is unknown; a number is true if it is not zero; a text is converted with the numeric prefix rule
  below. `WHERE`/`ON`/`HAVING` keep a row only when the condition is true (1), not for 0 or NULL.
* Numeric prefix rule (used wherever a text is used as a number, e.g. `'12abc' + 1`): the longest leading
  decimal number (optional white space, optional sign, digits with optional fraction and exponent) is used;
  an `int` if it has no `.` or exponent, else float; if there is none the value is 0.

## 4. Expressions

Operator precedence, highest first (operators in one row are left associative):

1. unary `-` `+`
2. `||`
3. `*` `/` `%`
4. `+` `-`
5. `<` `<=` `>` `>=`
6. `=` `==` `<>` `!=` `IS` `IS NOT` `IN` `NOT IN` `LIKE` `NOT LIKE` `BETWEEN` `NOT BETWEEN`
7. `NOT`
8. `AND`
9. `OR`

* Arithmetic `+ - *`: operands converted with the numeric prefix rule; int op int is int, otherwise float. NULL
  operand gives NULL. (Integer overflow beyond 64 bits is unspecified.)
* `/`: int/int is integer division truncating toward zero (`-7/2 = -3`); if either operand is float the result is
  float division. Division by zero (any types) gives NULL.
* `%`: operands are converted to int; result has the sign of the dividend (`-7 % 3 = -1`, `7 % -3 = 1`); modulo
  zero gives NULL.
* `||`: text concatenation; operands that are int are converted to their decimal text (float operands are
  unspecified); a NULL operand gives NULL.
* Comparison operators give 1, 0 or NULL (NULL if either side is NULL). `==` is a synonym of `=`, `!=` of `<>`.
* `IS` / `IS NOT` never give NULL: `NULL IS NULL` is 1, `1 IS NULL` is 0, `a IS b` is true when both are NULL or
  compare equal.
* `AND`/`OR`/`NOT` use three-valued logic (`NULL AND 0 = 0`, `NULL OR 1 = 1`, `NOT NULL = NULL`); results are 1/0/NULL.
* `x BETWEEN a AND b` is `x >= a AND x <= b`; `NOT BETWEEN` negates it (NULL stays NULL).
* `x IN (e1, e2, ...)`: 1 if some `ei` equals x; else NULL if x is NULL or any `ei` is NULL; else 0. `x IN ()`
  (empty list) is 0 even for NULL x. `NOT IN` is the three-valued negation. `x IN (SELECT ...)` uses the first
  (only) column of the subquery with the same rules.
* `x LIKE p`: `%` matches any sequence (also empty), `_` matches exactly one character, everything else matches
  itself; ASCII letters match case-insensitively, other characters exactly. No ESCAPE clause. NULL operand gives
  NULL. Non-text operands are converted to text with the `||` rule.
* `CASE WHEN c1 THEN r1 ... [ELSE e] END` and `CASE x WHEN v1 THEN r1 ... [ELSE e] END` (simple form: `x = vi`,
  NULL never matches). No matching branch and no ELSE gives NULL.
* `EXISTS (SELECT ...)` is 1 if the subquery returns at least one row, else 0 (never NULL).
* `(SELECT ...)` as an expression is a scalar subquery: it must have one column; it yields the first row's value
  or NULL when there is no row (extra rows are ignored). More than one column is a `syntax` error.

### Scalar functions

NULL arguments give NULL unless stated otherwise. Wrong argument counts and unknown functions are `syntax` errors.

| function | result |
|---|---|
| `abs(x)` | absolute value (int stays int, float stays float) |
| `length(s)` | number of characters (code points) |
| `lower(s)`, `upper(s)` | ASCII letters only are changed |
| `substr(s, start [, len])` | 1-based; defined only for `start >= 1` and `len >= 0` (others unspecified); result shorter if the text ends |
| `round(x [, n])` | always a float; rounds half away from zero to `n` decimals (`n >= 0`, default 0): `round(2.5) = 3.0`, `round(-2.5) = -3.0`, `round(2.25, 1) = 2.3`. Tests only use values whose decimal representation is exact in binary or clearly away from a tie. |
| `coalesce(a, b, ...)` | first non-NULL argument (at least 2 arguments); NULL if all are NULL (does not propagate NULL) |
| `ifnull(a, b)` | same as `coalesce(a, b)` |
| `nullif(a, b)` | NULL if `a = b` (both non-NULL, equal) else `a` |
| `typeof(x)` | `'null'`, `'integer'`, `'real'` or `'text'` (does not propagate NULL) |
| `min(a, b, ...)`, `max(a, b, ...)` | two or more arguments: smallest/largest by the comparison order; NULL if any argument is NULL (with ONE argument these are aggregates) |
| `replace(s, from, to)` | all non-overlapping occurrences replaced left to right; empty `from` returns `s` |
| `trim(s [, chars])`, `ltrim`, `rtrim` | strip characters from both/left/right; default set is the space character; `chars` is a set of characters |
| `instr(s, t)` | 1-based position of the first occurrence of `t` in `s`, 0 if absent, `instr(s, '') = 1` |

### Aggregates

`count(*)`, `count(x)` (non-NULL values), `count(DISTINCT x)`, `sum(x)`, `avg(x)`, `min(x)`, `max(x)` (one
argument); `DISTINCT` is allowed inside `count`, `sum`, `avg`, `min`, `max`. NULL values are skipped.
`sum` of no values is NULL (not 0), is an int if all inputs are ints, else a float. `avg` is a float (NULL when no
values). `min`/`max` use the comparison order and give NULL when no values. `count` is never NULL (0 when empty).
An aggregate outside a SELECT list / HAVING / ORDER BY (for example in WHERE) is a `syntax` error.

## 5. Statements

```
CREATE TABLE [IF NOT EXISTS] name ( coldef {, coldef} )
coldef := name type { PRIMARY KEY | NOT NULL | UNIQUE | DEFAULT literal }
type   := INTEGER | TEXT | REAL            (the type is required; anything else is a syntax error)
literal:= [-] number | string | NULL

DROP TABLE [IF EXISTS] name
INSERT INTO name [( col {, col} )] VALUES ( expr {, expr} ) {, ( ... )}
INSERT INTO name [( col {, col} )] select
UPDATE name SET col = expr {, col = expr} [WHERE expr]
DELETE FROM name [WHERE expr]
BEGIN | COMMIT | ROLLBACK
select
```

* At most one PRIMARY KEY column per table (more is a `syntax` error). A PRIMARY KEY column may not hold NULL and
  must be unique. An `INTEGER PRIMARY KEY` column additionally auto-fills: when an inserted row gives NULL or
  omits it, it receives `max(existing values) + 1` (1 for an empty table) -- also after deletes (the maximum is
  recomputed, as in SQLite without AUTOINCREMENT).
* `NOT NULL` rejects NULL on INSERT and UPDATE. `UNIQUE` rejects a second equal non-NULL value (several NULLs are
  fine). Equality uses the comparison order of section 3 after affinity conversion. Violations are `constraint` errors.
* Omitted INSERT columns get their DEFAULT, or NULL. `DEFAULT` literals are stored as given then converted by
  affinity. An INSERT with the wrong number of values for the column list is a `syntax` error. INSERT expressions
  cannot reference columns. Unknown column names in a column list or SET list are `no_column` errors.
* `rowcount` of INSERT is the number of inserted rows; UPDATE the number of rows matched by WHERE (even if the
  new values equal the old ones); DELETE the number of deleted rows.
* UPDATE evaluates all right-hand sides against the row as it was before the statement changed it; WHERE and SET
  expressions may contain subqueries (also correlated to the row being updated, referring to it by table name).
  DELETE likewise.
* Table scans (SELECT without ORDER BY) return rows in insertion order in a simple implementation, but the order
  is UNSPECIFIED: tests only compare unordered results as multisets unless the query has ORDER BY.
* `CREATE TABLE` of an existing name is an `exists` error (unless IF NOT EXISTS: then nothing happens);
  duplicate column names are a `syntax` error. `DROP TABLE` of a missing table is `no_table` (unless IF EXISTS).

## 6. SELECT

```
select := core { UNION [ALL] core } [ORDER BY term {, term}] [LIMIT expr [OFFSET expr]]
core   := SELECT [DISTINCT] item {, item}
          [FROM source { [INNER | LEFT [OUTER]] JOIN source ON expr }]
          [WHERE expr] [GROUP BY expr {, expr}] [HAVING expr]
item   := * | name.* | expr [[AS] alias]
source := name [[AS] alias] | ( select ) [[AS] alias]
term   := expr [ASC | DESC]
```

* Evaluation order: FROM/JOIN, WHERE, GROUP BY / aggregates, HAVING, select list, DISTINCT, UNION, ORDER BY, LIMIT.
* A table is referred to by its alias if it has one, else by its name. Column references are `name` or
  `table.name`. An unqualified name that matches columns of several sources in the same scope is an `ambiguous`
  error; no match anywhere (including outer queries) is `no_column`; an unknown table qualifier is `no_column`.
  `*` expands to all columns of all sources in order, `t.*` to those of one source (unknown t: `no_table`).
  `SELECT *` without FROM is a `syntax` error.
* A query without FROM produces exactly one row (if WHERE is true).
* `JOIN` is nested-loop `INNER JOIN`. `LEFT JOIN` keeps every left row; when no right row satisfies ON, the right
  columns are NULL. Only `ON` joins exist (no comma joins, USING or NATURAL). A parenthesised subquery in FROM is
  a derived table; its column names are the output names of its select list; the alias is optional.
* Output column names (`Result.columns`): the alias if given; else the column name for a plain column reference
  (as written in the table definition); else the exact source text of the expression (unspecified, not tested
  for such items). For `*`, the table's column names.
* GROUP BY takes expressions over the FROM columns (not aliases or positions). A query is an aggregate query if it
  has GROUP BY or any aggregate in its select list or HAVING. Without GROUP BY all rows form one group, which exists
  even if there are no rows (so `SELECT count(*), sum(x) FROM empty` returns one row `(0, NULL)`). With GROUP BY, an
  empty input gives no rows. Non-aggregated columns in the select list that are not GROUP BY expressions take
  their value from an arbitrary row of the group (tests avoid this). HAVING is evaluated per group and may use
  aggregates and group expressions (aliases in HAVING are unspecified).
* `DISTINCT` removes duplicate output rows (values equal under the comparison order, `1` and `1.0` are equal,
  NULLs are equal to each other). `UNION` removes duplicates over the whole result, `UNION ALL` keeps them; all
  cores must have the same number of columns (else `syntax`); column names come from the first core.
  Combined with DISTINCT/UNION, the order of the surviving rows is unspecified unless ORDER BY is given.
* ORDER BY terms are tried in this order: an integer literal `n` = the n-th output column (out of range is a
  `syntax` error); an unqualified name equal to an output column name/alias (case-insensitive); otherwise any
  expression over the FROM columns (aggregates allowed in aggregate queries). For `UNION` queries only the first two
  forms are allowed. Sorting is by the comparison order: NULL first for ASC (so NULL last for DESC); ties keep an
  unspecified order. Multiple terms sort lexicographically with a direction each.
* `LIMIT n [OFFSET m]`: constant integer expressions (may use `?`). A negative LIMIT means no limit; a negative
  OFFSET counts as 0; non-integers are a `syntax` error. Applied after ORDER BY.
* Subqueries (in `IN`, `EXISTS`, scalar position, FROM) may be correlated: inside a subquery, names that do not
  resolve against its own FROM sources resolve against the enclosing queries (innermost first). A correlated
  subquery is evaluated per outer row.

## 7. Transactions

* Without BEGIN each statement is its own transaction. `BEGIN` starts an explicit one; `COMMIT` makes it
  permanent; `ROLLBACK` discards every change made since BEGIN, including `CREATE TABLE`/`DROP TABLE`.
* `BEGIN` inside a transaction, `COMMIT`/`ROLLBACK` without one: `txn` error (the open transaction, if any, stays open).
* A failing statement inside a transaction undoes only itself; the transaction stays open.
* Reads inside the transaction see its uncommitted changes.

## 8. Error codes

`SQLError.code` is exactly one of:

| code | when |
|---|---|
| `syntax` | malformed SQL, unknown function, wrong argument or parameter count, column-count mismatches, misuse of aggregates, LIMIT not integer, ORDER BY position out of range, duplicate column names in CREATE, unsupported column type |
| `no_table` | unknown table (in FROM, INSERT, UPDATE, DELETE, DROP, `t.*`) |
| `no_column` | unknown column (anywhere, including INSERT column lists, SET lists, qualified names) |
| `ambiguous` | an unqualified column name matches several sources |
| `exists` | CREATE TABLE of an existing table |
| `constraint` | NOT NULL / PRIMARY KEY / UNIQUE violation |
| `txn` | BEGIN in a transaction, COMMIT or ROLLBACK outside one |
| `corrupt` | database file with a bad header or an invalid log line other than a torn last line (section 10) |

When a statement has several problems, the check order is: syntax (the whole statement is parsed first), then
table lookup, then column lookup / ambiguity during evaluation, then constraints. Tests do not depend on finer detail.

## 9. Example session

```python
db = minisql.Database()
db.execute("CREATE TABLE t(id INTEGER PRIMARY KEY, name TEXT NOT NULL, score REAL)")
db.execute("INSERT INTO t(name, score) VALUES ('a', 1), ('b', 2.5)").rowcount       # 2
db.execute("SELECT id, name, score FROM t ORDER BY id").rows                        # [(1, 'a', 1.0), (2, 'b', 2.5)]
db.execute("SELECT count(*), avg(score) FROM t").rows                               # [(2, 1.75)]
db.execute("SELECT 7/2, -7/2, 7%3, 'x'||1").rows                                    # [(3, -3, 1, 'x1')]
db.execute("INSERT INTO t(name) VALUES (?)", ("c",))
try: db.execute("INSERT INTO t(id, name) VALUES (1, 'dup')")
except minisql.SQLError as e: e.code                                                # 'constraint'
```

## 10. Persistence: the log file

`Database(path)` with a file path stores the database as an append-only text log named by `path`:

* Line 1 is exactly `MINISQL-LOG 1`. If the file does not exist (or is empty) it is created with that line.
* Every further line is one committed transaction: a JSON object `{"txn": [[sql, [params...]], ...]}` followed by
  `\n`, where each pair is the original SQL text and the parameter list of one successful state-changing statement
  (CREATE, DROP, INSERT, UPDATE, DELETE) of that transaction in order. The SQL text is stored exactly as passed to
  `execute`; params are JSON values (null, numbers, strings). Statements that fail, SELECT, BEGIN, COMMIT and
  ROLLBACK are never logged. A transaction with no logged statement writes no line.
* Autocommit: a successful statement outside BEGIN/COMMIT is logged immediately as its own one-statement line.
  Inside BEGIN..COMMIT the statements are buffered and written as ONE line at COMMIT. After ROLLBACK, or if the
  database is closed with the transaction still open, nothing is written for it. Each line is flushed and
  `os.fsync`-ed before `execute` returns.
* Opening an existing file replays every line in order, executing its statements with the same engine (the log is
  not written while replaying). The result must equal the state before closing. The first line must match the
  header, else `SQLError("corrupt")`.
* Torn tail: if the file does not end with `\n`, the text after the last `\n` is an incomplete (possibly valid
  looking) last line written by a crashed process. It is ignored and the file is truncated to end after the last
  `\n` before the database is used. Every other line must be valid JSON of the above shape, else `corrupt`.
* The grader checks both directions: files written by your `Database` are parsed and replayed by an independent
  implementation (SQLite itself), and files written by the reference implementation (including torn ones) must be
  readable by yours.

## 11. Command line

`python3 minisql.py DBFILE` (DBFILE `:memory:` for an in-memory database) reads all of standard input, splits it
into statements at every `;` that is outside single-quoted strings and double-quoted identifiers, ignores segments
that are empty or only white space, and executes the rest in order with one `Database`. For each statement:

* a SELECT prints one line per row: values joined by `|`; NULL as `NULL`; ints in decimal; floats with Python's
  `repr()` (`1.0`, `0.30000000000000004`, `1e+20`); text as is (newlines inside text are printed as they are).
* other statements print nothing.
* an error prints `ERROR <code>` (one line) to standard output and execution continues with the next statement.

After the last statement the database is closed; the process exits with status 0. If opening the file fails with
`SQLError`, print `ERROR <code>` and exit with status 1.

## 12. What is NOT required

Blobs, AUTOINCREMENT, indexes, views, triggers, ALTER TABLE, table-level constraints, CAST, `GROUP_CONCAT`, window
functions, date functions, `ESCAPE`, `GLOB`, `CROSS/NATURAL/RIGHT/FULL` joins, `USING`, `WITH`, comma joins,
`INSERT OR ...`, `EXCEPT`/`INTERSECT`, comments in SQL, multi-statement `execute`, concurrency, 64-bit integer overflow
behaviour, and performance beyond "tables of a few hundred rows answer in well under a second per statement".
