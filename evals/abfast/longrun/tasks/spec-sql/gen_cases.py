#!/usr/bin/env python3
"""Records hidden + public cases for spec-sql from the sqlite3 oracle.  gen_cases.py HIDDEN_DIR PUBLIC_DIR"""
import json, os, random, re, sqlite3, sys
H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(H, "src"))
import minisql  # only for fmt()/split_statements() formatting helpers of the CLI

HID, PUB = sys.argv[1], sys.argv[2]

FIX = [
 "CREATE TABLE dept(code TEXT PRIMARY KEY, title TEXT NOT NULL, budget INTEGER)",
 "CREATE TABLE emp(id INTEGER PRIMARY KEY, name TEXT NOT NULL, dept TEXT, salary INTEGER, boss INTEGER, rating REAL, city TEXT DEFAULT 'Oslo')",
 "CREATE TABLE proj(pid INTEGER PRIMARY KEY, title TEXT, owner INTEGER, hours INTEGER)",
 "INSERT INTO dept VALUES ('eng','Engineering',1000),('ops','Operations',500),('sales','Sales',300),('hr','People',200),('fin','Finance',NULL)",
 "INSERT INTO emp VALUES (1,'Alice','eng',120,NULL,4.5,'Oslo'),(2,'Bob','eng',100,1,3.5,'Bergen'),(3,'Carol','eng',100,1,4.0,'Oslo'),"
 "(4,'Dave','ops',80,1,3.0,'Oslo'),(5,'Erin','ops',90,4,NULL,'Bergen'),(6,'Frank','sales',70,1,2.5,'Tromso'),(7,'Grace','sales',75,6,3.5,'Oslo'),"
 "(8,'Heidi',NULL,60,1,NULL,'Bergen')",
 "INSERT INTO emp VALUES (9,'Ivan','eng',NULL,2,2.0,'Tromso'),(10,'Judy','hr',85,1,4.5,'Oslo'),(11,'Karl','ops',80,4,3.0,'Bergen'),(12,'Liam','sales',70,6,NULL,'Oslo')",
 "INSERT INTO proj VALUES (1,'Apollo',1,40),(2,'Borealis',2,25),(3,'Cobalt',2,30),(4,'Delta',3,NULL),(5,'Echo',NULL,10),(6,'Foxtrot',7,55),(7,'Gamma',4,20),(8,'Helix',12,5)",
]


def code_of(e):
    m = str(e).lower()
    if isinstance(e, sqlite3.IntegrityError):
        return "constraint"
    if "no such table" in m: return "no_table"
    if "no such column" in m or "has no column named" in m: return "no_column"
    if "already exists" in m: return "exists"
    if "ambiguous" in m: return "ambiguous"
    if "transaction" in m: return "txn"
    if any(s in m for s in ("syntax error", "unrecognized token", "incomplete input", "misuse of aggregate", "no such function",
                            "values for", "values were supplied", "wrong number of arguments", "no tables specified", "order by term")):
        return "syntax"
    raise RuntimeError(f"unmapped sqlite error: {e!r}")


class Oracle:
    def __init__(self):
        self.con = sqlite3.connect(":memory:", isolation_level=None)

    def step(self, sql, params=(), cols=False, ordered=None):
        try:
            cur = self.con.execute(sql, params)
        except sqlite3.Error as e:
            return {"err": code_of(e)}
        if cur.description is not None:
            rows = [list(r) for r in cur.fetchall()]
            ex = {"rows": rows, "ordered": bool(re.search(r"order\s+by", sql, re.I)) if ordered is None else ordered}
            if cols:
                ex["cols"] = [d[0] for d in cur.description]
            return ex
        return {"rowcount": cur.rowcount if re.match(r"\s*(insert|update|delete)", sql, re.I) else 0}

    def close(self):
        if self.con.in_transaction:
            self.con.execute("ROLLBACK")


cases = []


def mk(ms, steps, fixture=True, kind="steps", **extra):
    o = Oracle()
    out = []
    pre = []
    if fixture:
        for s in FIX:
            o.con.execute(s)
            pre.append({"sql": s, "params": [], "expect": {"rowcount": 0 if not s.startswith("INSERT") else None}})
    for s in steps:
        if isinstance(s, str):
            s = (s, ())
        sql, params = s[0], (s[1] if len(s) > 1 else ())
        kw = s[2] if len(s) > 2 else {}
        out.append({"sql": sql, "params": list(params), "expect": o.step(sql, params, **kw)})
    for p in pre:
        e = p["expect"]
        if e.get("rowcount") is None:
            e.pop("rowcount")
    c = {"id": f"{ms}-{len([x for x in cases if x['ms'] == ms]):03d}", "ms": ms, "kind": kind, "steps": pre + out}
    c.update(extra)
    cases.append(c)
    return c


def Q(ms, sqls):
    for s in sqls:
        mk(ms, [s])


# ---- s01 expressions without tables
Q("s01", [
 "SELECT 1+2*3", "SELECT 10/3, 10%3, -10/3, -10%3, 10/-3, 10%-3", "SELECT 7.5/2, 1/2.0, 3*1.5, 2-0.5",
 "SELECT 5/0, 5%0, 5.0/0", "SELECT 'a'||'b'||'c'", "SELECT 1<2, 2<1, 1=1.0, 'a'<'b', 'a'<'B', 'abc'='abc', 2<>3, 2!=2, 3>=3, 3<=2",
 "SELECT 1 AND 0, 1 OR NULL, NULL AND 0, NULL OR 0, NOT NULL, NOT 5, 0 OR 0, NULL AND 1, NULL OR 1",
 "SELECT NULL=NULL, NULL IS NULL, 1 IS NOT NULL, NULL IS 1, 1 IS 1, 1 IS NOT 2",
 "SELECT 2 IN (1,2,3), 5 NOT IN (1,2), NULL IN (1,2), 3 IN (1,NULL), 3 NOT IN (1,NULL), 1 IN (), 2 IN (2,NULL)",
 "SELECT 5 BETWEEN 1 AND 10, 5 NOT BETWEEN 6 AND 8, 5 BETWEEN 5 AND 5, NULL BETWEEN 1 AND 2",
 "SELECT 'abc' LIKE 'A%', 'abc' LIKE 'a_c', 'abc' LIKE 'b%', 'abc' NOT LIKE 'x%', 'abc' LIKE '%', '' LIKE '%', 'a.c' LIKE 'a.c', NULL LIKE 'a'",
 "SELECT CASE WHEN 1>2 THEN 'x' WHEN 2>1 THEN 'y' END, CASE WHEN 0 THEN 1 END, CASE WHEN NULL THEN 1 ELSE 2 END",
 "SELECT CASE 3 WHEN 1 THEN 'a' WHEN 3 THEN 'c' ELSE 'z' END, CASE 9 WHEN 1 THEN 'a' END, CASE NULL WHEN NULL THEN 1 ELSE 0 END",
 "SELECT -(-3), +4, - -2, -(1+2)", "SELECT 1+NULL, 'a'||NULL, NULL*0, NULL||NULL", "SELECT 2*3+4*5-6/3", "SELECT (1+2)*3, ((4))",
 "SELECT 1 < 2 = 1, 2 = 2 = 1", "SELECT NOT 1 = 2, NOT 0 OR 1", "SELECT 1+2 || 3, 2*3 || 4", "SELECT 1 OR 0 AND 0, (1 OR 0) AND 0",
 "SELECT '3'+4, '3.5'*2, 'abc'+1, '12abc'+1, ' 7'+0", "SELECT 9223372036854775807, -9223372036854775807", "SELECT 1e3, 2.5e-1, .5, 3.",
 "SELECT typeof(1), typeof(1.0), typeof('a'), typeof(NULL), typeof(1+1.0), typeof(1/2), typeof(1/2.0), typeof(1=1)",
 "SELECT 'it''s', 'a''''b', length('it''s')", "SELECT 7 % 4 * 2, 100 / 10 / 5, 100 - 10 - 5", "SELECT 1 = 1 AND 2 = 2, 1 = 2 OR 3 = 3",
 "SELECT 5 > 3 > 1, 1 < 2 < 3", "SELECT 0.1 + 0.2, 1.5 * 2, 10 / 4.0, 2 * 0.5", "SELECT 1 WHERE 1", "SELECT 1 WHERE 0", "SELECT 1 AS a, 2 AS b, 3 c",
 "SELECT 'x' = 'X', 'x' < 'y', 'Z' < 'a', '' < 'a', 'a' < 'ab'", "SELECT 3 BETWEEN 1+1 AND 2*2, 2 IN (1+1, 3)",
 "SELECT 1 = 'a', 1 < 'a', 'a' > 5, 5 < 'abc'",
])
for sql, p in [("SELECT ?+?", (2, 3)), ("SELECT ? || ?", ("a", "b")), ("SELECT ? IS NULL", (None,)), ("SELECT ?, ?, ?", (1, 2.5, "x")),
               ("SELECT ? * 2 + ?", (4, 1)), ("SELECT ? IN (1,2,3)", (2,)), ("SELECT ? LIKE 'a%'", ("Apple",))]:
    mk("s01", [(sql, p)])

# ---- s02 DDL / INSERT / basic SELECT
Q("s02", [
 "SELECT * FROM dept", "SELECT id, name FROM emp", "SELECT name AS n, salary s FROM emp", "SELECT e.name FROM emp e", "SELECT emp.name, emp.id FROM emp",
 "SELECT * FROM proj", "SELECT city FROM emp WHERE id = 1", "SELECT id, city FROM emp WHERE id = 12", "SELECT code, budget FROM dept",
 "SELECT e.* FROM emp e WHERE e.id < 3", "SELECT emp.* FROM emp WHERE id = 5", "SELECT salary * 2, name FROM emp WHERE id < 4",
 "SELECT name || ' (' || dept || ')' FROM emp WHERE id < 4", "SELECT DEPT, Name FROM EMP WHERE ID = 2",
 "SELECT t.id FROM (SELECT id FROM emp WHERE id < 4) t", "SELECT 1 FROM emp WHERE id = 1", "SELECT typeof(rating), typeof(salary), typeof(name) FROM emp WHERE id = 1",
])
mk("s02", ["CREATE TABLE a(x INTEGER PRIMARY KEY, y TEXT DEFAULT 'd', z REAL)", "INSERT INTO a(y) VALUES ('p')", "INSERT INTO a(x,z) VALUES (10, 5),(11,2)",
           "INSERT INTO a VALUES (NULL,'q',1.5)", "SELECT * FROM a ORDER BY x", "SELECT typeof(z), z FROM a ORDER BY x"], fixture=False)
mk("s02", ["CREATE TABLE b(n INTEGER, t TEXT, r REAL)", "INSERT INTO b VALUES (5.0, 42, 3), ('7', 'x', '2.5')", "SELECT n, t, r, typeof(n), typeof(t), typeof(r) FROM b"], fixture=False)
mk("s02", ["CREATE TABLE IF NOT EXISTS c(v INTEGER)", "CREATE TABLE IF NOT EXISTS c(v INTEGER)", "INSERT INTO c VALUES (1),(2),(3)", "SELECT sum(v) FROM c"], fixture=False)
mk("s02", ["CREATE TABLE d(k INTEGER PRIMARY KEY, v TEXT)", "INSERT INTO d(v) VALUES ('a'),('b')", "INSERT INTO d VALUES (100,'c')", "INSERT INTO d(v) VALUES ('d')",
           "SELECT k, v FROM d ORDER BY k"], fixture=False)
mk("s02", ["CREATE TABLE e(a INTEGER DEFAULT 7, b TEXT DEFAULT 'zz', c REAL DEFAULT -1.5, d INTEGER)", "INSERT INTO e(d) VALUES (1)", "INSERT INTO e VALUES (NULL, NULL, NULL, 2)",
           "SELECT * FROM e ORDER BY d"], fixture=False)
mk("s02", [("INSERT INTO dept VALUES (?,?,?)", ("it", "I.T.", 50)), ("INSERT INTO dept(code,title) VALUES (?,?)", ("qa", "Quality")), ("SELECT * FROM dept WHERE code IN ('it','qa')",)])
mk("s02", ["INSERT INTO emp(id, name) VALUES (20,'Zed'),(21,'Yan')", "SELECT id,name,dept,salary,city FROM emp WHERE id >= 20 ORDER BY id"])
mk("s02", ["CREATE TABLE \"my table\"(\"a b\" INTEGER, c TEXT)", "INSERT INTO \"my table\" VALUES (1,'x')", "SELECT \"a b\", c FROM \"my table\""], fixture=False)
mk("s02", ["CREATE TABLE s(v TEXT)", "INSERT INTO s VALUES ('multi word'), ('quo''te'), (''), ('ünï'), ('日本')", "SELECT v, length(v) FROM s ORDER BY v"], fixture=False)
mk("s02", ["SELECT   id ,name\n FROM\temp\nWHERE id=1 ;"])
mk("s02", ["select ID, NAME from Emp where Id = 3"])
mk("s02", ["SELECT id, name FROM emp WHERE id = 1", "select id from emp where id = 2;"], )

# ---- s03 WHERE
w = ["salary > 80", "salary >= 100 AND dept='eng'", "dept='ops' OR dept='hr'", "NOT dept='eng'", "dept IS NULL", "dept IS NOT NULL AND salary IS NULL",
     "salary IN (70,80)", "salary NOT IN (70,80)", "dept IN ('eng','hr')", "dept NOT IN ('eng')", "salary BETWEEN 70 AND 90", "salary NOT BETWEEN 70 AND 90",
     "name LIKE 'A%'", "name LIKE '%e'", "name LIKE '_a%'", "name NOT LIKE '%a%'", "city = 'Oslo' AND (dept='ops' OR salary>100)", "salary > 80 AND NOT (dept = 'eng')",
     "rating > 3.0", "rating IS NULL", "salary * 2 > 150", "salary - boss > 50", "boss = 1", "boss <> 1", "boss != 4", "name = 'alice'", "1=1", "1=0", "salary = NULL",
     "NOT (salary > 80)", "dept = 'eng' AND salary IS NOT NULL OR city = 'Tromso'", "id % 2 = 0", "salary / 3 = 30", "name LIKE 'a%'", "name LIKE '%'",
     "rating >= 3.5 AND rating <= 4.0", "(dept = 'eng' OR dept = 'ops') AND salary < 100", "NOT (dept IS NULL OR salary IS NULL)", "salary IN (SELECT 80)",
     "id BETWEEN 3 AND 5 OR id > 10", "name > 'H' AND name < 'M'", "city IN ('Oslo') AND boss IS NOT NULL AND rating IS NOT NULL", "salary IS NULL OR salary < 70",
     "dept = 'eng' AND (salary > 100 OR salary IS NULL)", "NOT salary", "salary - 100", "boss IN (1, 4) AND NOT city = 'Oslo'"]
Q("s03", [f"SELECT id FROM emp WHERE {c}" for c in w])

# ---- s04 order / limit / distinct
Q("s04", [
 "SELECT name FROM emp ORDER BY name", "SELECT id FROM emp ORDER BY salary, id", "SELECT id FROM emp ORDER BY salary DESC, id", "SELECT id FROM emp ORDER BY dept, salary DESC, id",
 "SELECT name, salary FROM emp ORDER BY 2 DESC, 1", "SELECT name AS n FROM emp ORDER BY n DESC", "SELECT id FROM emp ORDER BY rating DESC, id",
 "SELECT id FROM emp ORDER BY id LIMIT 3", "SELECT id FROM emp ORDER BY id LIMIT 5 OFFSET 3", "SELECT id FROM emp ORDER BY id LIMIT 0",
 "SELECT id FROM emp ORDER BY id LIMIT 100 OFFSET 10", "SELECT id FROM emp ORDER BY id LIMIT -1 OFFSET 8", "SELECT DISTINCT dept FROM emp",
 "SELECT DISTINCT dept, city FROM emp", "SELECT DISTINCT salary FROM emp ORDER BY salary", "SELECT DISTINCT dept FROM emp ORDER BY dept DESC",
 "SELECT id FROM emp ORDER BY salary * -1, id", "SELECT name FROM emp ORDER BY length(name), name", "SELECT id FROM emp ORDER BY CASE WHEN dept='eng' THEN 0 ELSE 1 END, id",
 "SELECT name FROM emp ORDER BY salary DESC, id LIMIT 3", "SELECT id FROM emp WHERE dept='eng' ORDER BY id DESC", "SELECT id FROM emp ORDER BY city, name",
 "SELECT id FROM emp ORDER BY id LIMIT 2+1", "SELECT name FROM emp ORDER BY name LIMIT 3 OFFSET 2", "SELECT id, dept FROM emp ORDER BY dept DESC, id DESC",
 "SELECT DISTINCT city FROM emp ORDER BY city", "SELECT DISTINCT rating FROM emp ORDER BY rating", "SELECT id FROM emp ORDER BY boss, id", "SELECT id FROM emp ORDER BY id LIMIT 3 OFFSET 20",
 "SELECT name, salary AS pay FROM emp ORDER BY pay, name", "SELECT id FROM emp ORDER BY dept IS NULL, dept, id", "SELECT DISTINCT boss FROM emp WHERE boss IS NOT NULL ORDER BY boss DESC LIMIT 2",
 ("SELECT id FROM emp ORDER BY id LIMIT ? OFFSET ?", (4, 2)),
])

# ---- s05 functions
Q("s05", [
 "SELECT abs(-5), abs(5), abs(-2.5), abs(0), abs(NULL)", "SELECT length('hello'), length(''), length('日本語'), length(NULL)", "SELECT lower('HeLLo'), upper('HeLLo'), upper(NULL), lower('ÀB')",
 "SELECT substr('hello',2,3), substr('hello',3), substr('hello',1,1), substr('hello',5,10), substr('hello',6), substr('hello',2,0), substr('日本語',2,1)",
 "SELECT round(2.5), round(3.5), round(-2.5), round(0.5), round(1.4), round(-0.5), round(7)", "SELECT round(3.14159,2), round(1234.5678,1), round(2.25,1), round(5.5,0), round(1.005,0)",
 "SELECT coalesce(NULL,NULL,3), coalesce(1,2), coalesce(NULL,'x'), ifnull(NULL,'x'), ifnull(5,6)", "SELECT nullif(1,1), nullif(1,2), nullif('a','a'), nullif(NULL,1)",
 "SELECT typeof(abs(-1)), typeof(round(1)), typeof(length('a')), typeof(upper('a')), typeof(coalesce(NULL,1.5))", "SELECT min(3,1,2), max(3,1,2), max('a','b'), min(1,NULL), max(1.5, 1)",
 "SELECT replace('banana','an','AN'), replace('abc','','x'), replace('aaa','a','bb'), replace('abc','z','y')", "SELECT trim('  hi  '), ltrim('  hi  '), rtrim('  hi  '), trim('xxhixx','x'), ltrim('xxhixx','x'), rtrim('xxhixx','x')",
 "SELECT instr('hello','ll'), instr('hello','z'), instr('hello',''), instr('abcabc','c')",
 "SELECT upper(name), length(name) FROM emp WHERE id<4", "SELECT id, coalesce(dept,'none') FROM emp", "SELECT id, coalesce(salary,0)+1 FROM emp",
 "SELECT substr(name,1,1)||substr(lower(name),2) FROM emp WHERE id<=3", "SELECT id, round(rating) FROM emp", "SELECT id, round(rating*2)/2 FROM emp WHERE rating IS NOT NULL",
 "SELECT name FROM emp WHERE length(name) = 4", "SELECT name FROM emp WHERE upper(city) = 'OSLO' AND lower(name) LIKE 'a%'", "SELECT id, nullif(dept,'eng') FROM emp WHERE id < 6",
 "SELECT id, ifnull(boss, -1), typeof(boss) FROM emp WHERE id < 4", "SELECT id, abs(salary - 90) FROM emp", "SELECT id, max(salary, 85), min(salary, 85) FROM emp WHERE id < 5",
 "SELECT replace(name,'a','4'), instr(name,'a') FROM emp WHERE id < 5", "SELECT coalesce(dept, city, 'none'), id FROM emp WHERE id IN (8, 9)",
 "SELECT upper(substr(name,1,3)) || '-' || id FROM emp WHERE id < 4", "SELECT length(trim('  ' || name || '  ')) FROM emp WHERE id = 2",
 "SELECT round(avg(salary), 1), round(avg(rating), 2) FROM emp", "SELECT typeof(round(2.5, 1)), typeof(abs(-1.5)), typeof(nullif(1, 2))",
])

# ---- s06 aggregates
Q("s06", [
 "SELECT count(*) FROM emp", "SELECT count(dept), count(salary), count(rating), count(*) FROM emp", "SELECT count(DISTINCT dept), count(DISTINCT city) FROM emp", "SELECT sum(salary) FROM emp",
 "SELECT avg(salary) FROM emp", "SELECT min(salary), max(salary), min(name), max(name) FROM emp", "SELECT sum(rating), avg(rating) FROM emp", "SELECT count(*), sum(salary), avg(salary), min(salary), max(salary) FROM emp WHERE dept='nonexistent'",
 "SELECT sum(salary)+1, max(salary)-min(salary) FROM emp", "SELECT count(*) FROM emp WHERE salary>80", "SELECT sum(DISTINCT salary), count(DISTINCT salary) FROM emp", "SELECT avg(salary*2) FROM emp",
 "SELECT count(*) FROM proj WHERE hours IS NULL", "SELECT sum(hours), avg(hours), min(hours), max(hours) FROM proj", "SELECT min(rating), max(rating) FROM emp", "SELECT typeof(sum(salary)), typeof(avg(salary)), typeof(sum(rating)), typeof(count(*))  FROM emp",
 "SELECT sum(salary) / count(*), sum(salary) * 1.0 / count(*) FROM emp", "SELECT count(*) AS n FROM emp WHERE city = 'Oslo'", "SELECT max(length(name)), min(length(name)) FROM emp", "SELECT sum(budget), count(budget) FROM dept",
 "SELECT coalesce(sum(hours), 0) FROM proj WHERE owner = 99", "SELECT count(*) + 1 FROM emp", "SELECT CASE WHEN count(*) > 10 THEN 'many' ELSE 'few' END FROM emp", "SELECT min(dept), max(dept) FROM emp",
 "SELECT count(boss), count(DISTINCT boss) FROM emp", "SELECT sum(salary) FROM emp WHERE salary IS NULL", "SELECT avg(rating) FROM emp WHERE city = 'Nowhere'", "SELECT count(*) FROM emp, proj WHERE 1 = 0" if False else "SELECT count(*) FROM dept WHERE budget IS NULL",
])

# ---- s07 group by
Q("s07", [
 "SELECT dept, count(*) FROM emp GROUP BY dept", "SELECT dept, sum(salary), avg(salary) FROM emp GROUP BY dept", "SELECT dept, count(*) FROM emp GROUP BY dept HAVING count(*) > 2",
 "SELECT dept, sum(salary) FROM emp GROUP BY dept HAVING sum(salary) >= 200 ORDER BY dept", "SELECT city, dept, count(*) FROM emp GROUP BY city, dept",
 "SELECT dept, max(salary)-min(salary) FROM emp GROUP BY dept", "SELECT dept, count(*) FROM emp WHERE salary IS NOT NULL GROUP BY dept",
 "SELECT boss, count(*) c FROM emp GROUP BY boss ORDER BY c DESC, boss", "SELECT dept, min(name), max(name) FROM emp GROUP BY dept", "SELECT dept, count(DISTINCT city) FROM emp GROUP BY dept",
 "SELECT dept FROM emp GROUP BY dept HAVING max(salary) > 100", "SELECT dept, count(rating) FROM emp GROUP BY dept", "SELECT length(name), count(*) FROM emp GROUP BY length(name)",
 "SELECT dept, sum(salary)*1.0/count(*) FROM emp GROUP BY dept", "SELECT dept, CASE WHEN count(*)>2 THEN 'big' ELSE 'small' END FROM emp GROUP BY dept",
 "SELECT city, avg(rating) FROM emp GROUP BY city ORDER BY city", "SELECT dept, count(*) FROM emp GROUP BY dept ORDER BY count(*) DESC, dept", "SELECT count(*) FROM emp GROUP BY dept",
 "SELECT dept, sum(salary) FROM emp GROUP BY dept HAVING dept IS NOT NULL", "SELECT e.dept, count(*) FROM emp e GROUP BY e.dept", "SELECT dept, sum(salary) total FROM emp GROUP BY dept ORDER BY total DESC, dept",
 "SELECT city, count(*) FROM emp GROUP BY city HAVING count(*) >= 3 AND city <> 'Tromso'", "SELECT owner, count(*), sum(hours) FROM proj GROUP BY owner ORDER BY owner", "SELECT salary > 80, count(*) FROM emp GROUP BY salary > 80",
 "SELECT dept, group_total FROM (SELECT dept, sum(salary) AS group_total FROM emp GROUP BY dept) ORDER BY dept" if False else "SELECT dept, round(avg(salary)) FROM emp GROUP BY dept ORDER BY dept",
 "SELECT dept, count(*) n FROM emp GROUP BY dept HAVING count(*) > 1 ORDER BY n, dept", "SELECT boss, min(salary), max(rating) FROM emp GROUP BY boss ORDER BY boss",
])

# ---- s08 update / delete
dml = [
 "UPDATE emp SET salary = salary + 10 WHERE dept='eng'", "UPDATE emp SET city='Paris'", "UPDATE emp SET salary = NULL WHERE id=1", "UPDATE emp SET dept = 'x', city = 'y' WHERE salary > 100",
 "UPDATE emp SET salary = (SELECT max(salary) FROM emp) WHERE id=2", "DELETE FROM emp WHERE dept = 'ops'", "DELETE FROM emp", "DELETE FROM emp WHERE id > 100", "UPDATE emp SET rating = rating * 2",
 "UPDATE emp SET name = name || '!' WHERE id <= 2", "DELETE FROM proj WHERE hours IS NULL OR hours < 10", "UPDATE emp SET salary = salary * 2, rating = 1.5 WHERE city = 'Tromso'",
 "UPDATE emp SET boss = id WHERE boss IS NULL", "DELETE FROM emp WHERE salary IS NULL", "UPDATE dept SET budget = budget / 2", "UPDATE emp SET dept = NULL", "UPDATE emp SET salary = salary + 1 WHERE 0",
 "DELETE FROM emp WHERE id IN (SELECT owner FROM proj)", "UPDATE emp SET city = upper(city) WHERE city LIKE 'o%'", "UPDATE emp SET salary = CASE WHEN salary > 90 THEN 90 ELSE salary + 5 END",
 ("UPDATE emp SET salary = ? WHERE id = ?", (999, 3)), ("DELETE FROM emp WHERE dept = ? AND salary < ?", ("sales", 72)),
]
for s in dml:
    sql = s if isinstance(s, str) else s[0]
    tbl = re.search(r"(?:UPDATE|FROM)\s+(\w+)", sql).group(1)
    mk("s08", [s, f"SELECT * FROM {tbl} ORDER BY 1"])
mk("s08", ["INSERT INTO proj SELECT pid + 100, title, owner, hours FROM proj WHERE pid < 3", "SELECT count(*) FROM proj", "SELECT pid FROM proj ORDER BY pid"])
mk("s08", ["UPDATE emp SET salary = 1 WHERE id = 1", "UPDATE emp SET salary = 2 WHERE id = 1", "DELETE FROM emp WHERE id = 1", "DELETE FROM emp WHERE id = 1", "SELECT count(*) FROM emp"])
mk("s08", ["DELETE FROM proj", "INSERT INTO proj(title) VALUES ('new1'),('new2')", "SELECT pid, title FROM proj ORDER BY pid"])
mk("s08", ["DROP TABLE proj", "SELECT * FROM proj", "CREATE TABLE proj(a INTEGER)", "INSERT INTO proj VALUES (7)", "SELECT * FROM proj"])

# ---- s09 constraints & errors
errs = [
 "INSERT INTO emp(id,name) VALUES (1,'X')", "INSERT INTO emp(id) VALUES (50)", "INSERT INTO dept VALUES ('eng','Dup',1)", "INSERT INTO dept VALUES ('x', NULL, 1)",
 "SELECT * FROM nope", "SELECT nope FROM emp", "SELECT FROM emp", "SELEC 1", "CREATE TABLE emp(x INTEGER)", "DROP TABLE nope", "DROP TABLE IF EXISTS nope", "INSERT INTO nope VALUES (1)",
 "INSERT INTO emp(zzz) VALUES (1)", "SELECT title FROM dept d JOIN proj p ON 1=1", "UPDATE emp SET zzz=1", "DELETE FROM nope", "SELECT 1 +", "SELECT (1", "SELECT 'abc",
 "SELECT count(*) FROM emp WHERE count(*) > 1", "INSERT INTO emp(id,name) VALUES (1)", "SELECT nosuchfunc(1)", "SELECT x.id FROM emp", "SELECT emp.nope FROM emp",
 "UPDATE emp SET name = NULL WHERE id = 1", "UPDATE emp SET id = 1 WHERE id = 2", "INSERT INTO proj(pid, title) VALUES (1, 'dup')", "SELECT * FROM emp WHERE", "SELECT * FROM", "INSERT INTO emp",
 "SELECT 1 FROM emp GROUP", "CREATE TABLE", "SELECT id FROM emp ORDER", "DROP emp", "SELECT * FROM emp e, ", "INSERT INTO emp VALUES (1, 'a')" if False else "UPDATE emp SET",
 "SELECT id FROM emp e JOIN emp b ON e.boss = b.id", "SELECT max(1,2,3) FROM nope",
]
Q("s09", errs)
mk("s09", ["CREATE TABLE u(a INTEGER UNIQUE, b TEXT)", "INSERT INTO u VALUES (1,'x')", "INSERT INTO u VALUES (1,'y')", "INSERT INTO u VALUES (NULL,'n'),(NULL,'m')", "SELECT * FROM u ORDER BY b"], fixture=False)
mk("s09", ["INSERT INTO dept VALUES ('zz','Z',1),('eng','Dup',2),('yy','Y',3)", "SELECT code FROM dept ORDER BY code"])
mk("s09", ["UPDATE emp SET id = id + 1 WHERE id < 3", "SELECT id FROM emp ORDER BY id LIMIT 4"])
mk("s09", ["UPDATE emp SET salary = 1, name = NULL WHERE id < 3", "SELECT id, salary, name FROM emp WHERE id < 4 ORDER BY id"])
mk("s09", ["INSERT INTO emp(id,name) VALUES (30,'A'),(31,NULL)", "SELECT count(*) FROM emp"])
mk("s09", ["CREATE TABLE t(a INTEGER PRIMARY KEY)", "INSERT INTO t VALUES (5)", "INSERT INTO t VALUES (NULL)", "INSERT INTO t VALUES (3), (NULL)", "SELECT a FROM t ORDER BY a"], fixture=False)
mk("s09", ["CREATE TABLE t(a INTEGER NOT NULL DEFAULT 5, b TEXT)", "INSERT INTO t(b) VALUES ('x')", "INSERT INTO t VALUES (NULL, 'y')", "SELECT * FROM t"], fixture=False)
mk("s09", ["CREATE TABLE p(name TEXT PRIMARY KEY, v INTEGER)", "INSERT INTO p VALUES ('a',1)", "INSERT INTO p VALUES ('a',2)", "INSERT INTO p VALUES ('b',2)", "SELECT * FROM p ORDER BY name"], fixture=False)
mk("s09", ["SELECT * FROM emp", "SELECT * FROM nope", "SELECT id FROM emp WHERE id = 1"])
mk("s09", ["INSERT INTO emp(id,name,salary) VALUES (40,'E',(SELECT max(salary) FROM emp))", "SELECT salary FROM emp WHERE id = 40"])

# ---- s10 transactions
mk("s10", ["BEGIN", "INSERT INTO dept VALUES ('t1','T',1)", "SELECT count(*) FROM dept", "ROLLBACK", "SELECT count(*) FROM dept"])
mk("s10", ["BEGIN", "UPDATE emp SET salary = 0", "COMMIT", "SELECT sum(salary) FROM emp"])
mk("s10", ["BEGIN", "BEGIN", "ROLLBACK"])
mk("s10", ["COMMIT"])
mk("s10", ["ROLLBACK"])
mk("s10", ["BEGIN", "CREATE TABLE x(a INTEGER)", "INSERT INTO x VALUES (1)", "ROLLBACK", "SELECT * FROM x"])
mk("s10", ["BEGIN", "DROP TABLE proj", "SELECT * FROM proj", "ROLLBACK", "SELECT count(*) FROM proj"])
mk("s10", ["BEGIN", "INSERT INTO dept VALUES ('t1','T',1)", "INSERT INTO dept VALUES ('t1','dup',2)", "COMMIT", "SELECT code FROM dept ORDER BY code"])
mk("s10", ["BEGIN", "DELETE FROM emp", "ROLLBACK", "BEGIN", "DELETE FROM emp WHERE id > 6", "COMMIT", "SELECT count(*) FROM emp"])
mk("s10", ["BEGIN", "INSERT INTO dept VALUES ('a1','A',1)", "COMMIT", "COMMIT", "SELECT count(*) FROM dept"])
mk("s10", ["BEGIN", "UPDATE emp SET name='Z' WHERE id=1", "ROLLBACK", "BEGIN", "UPDATE emp SET name='Y' WHERE id=1", "COMMIT", "SELECT name FROM emp WHERE id=1"])
mk("s10", ["BEGIN", "INSERT INTO emp(id,name) VALUES (60,'n')", "SELECT name FROM emp WHERE id = 60", "UPDATE emp SET name='m' WHERE id=60", "SELECT name FROM emp WHERE id=60", "ROLLBACK", "SELECT name FROM emp WHERE id=60"])
mk("s10", ["INSERT INTO dept VALUES ('q','Q',1)", "BEGIN", "ROLLBACK", "SELECT count(*) FROM dept"])
mk("s10", ["BEGIN", "SELECT 1", "COMMIT", "BEGIN", "ROLLBACK", "BEGIN", "COMMIT"])
mk("s10", ["BEGIN", "CREATE TABLE y(a INTEGER)", "INSERT INTO y VALUES (1),(2)", "COMMIT", "BEGIN", "DELETE FROM y WHERE a = 1", "ROLLBACK", "SELECT count(*) FROM y"])

# ---- s11 joins
Q("s11", [
 "SELECT e.name, d.title FROM emp e JOIN dept d ON e.dept = d.code", "SELECT e.name, d.title FROM emp e LEFT JOIN dept d ON e.dept = d.code",
 "SELECT d.title, count(e.id) FROM dept d LEFT JOIN emp e ON e.dept = d.code GROUP BY d.title", "SELECT e.name, p.title FROM emp e JOIN proj p ON p.owner = e.id ORDER BY e.id, p.pid",
 "SELECT e.name, d.title, p.title FROM emp e JOIN dept d ON e.dept = d.code JOIN proj p ON p.owner = e.id", "SELECT e.name, b.name FROM emp e JOIN emp b ON e.boss = b.id",
 "SELECT e.name, b.name FROM emp e LEFT JOIN emp b ON e.boss = b.id", "SELECT e.name FROM emp e LEFT JOIN proj p ON p.owner = e.id WHERE p.pid IS NULL",
 "SELECT d.code, sum(e.salary) FROM dept d LEFT JOIN emp e ON e.dept = d.code GROUP BY d.code", "SELECT e.name, d.title FROM emp e INNER JOIN dept d ON e.dept = d.code AND d.budget > 400",
 "SELECT e.name, d.title FROM emp e LEFT JOIN dept d ON e.dept = d.code AND d.budget > 400", "SELECT e.name, d.title FROM emp e LEFT OUTER JOIN dept d ON e.dept = d.code WHERE d.title IS NULL",
 "SELECT x.dept, x.n FROM (SELECT dept, count(*) n FROM emp GROUP BY dept) x WHERE x.n > 2", "SELECT a.id, b.id FROM emp a JOIN emp b ON a.salary < b.salary AND a.dept = b.dept WHERE a.id < 4",
 "SELECT * FROM dept d JOIN proj p ON p.pid = 1", "SELECT d.*, p.pid FROM dept d JOIN proj p ON p.pid = 1 AND d.code = 'eng'", "SELECT p.title, e.name FROM proj p LEFT JOIN emp e ON p.owner = e.id",
 "SELECT count(*) FROM emp e JOIN dept d ON e.dept = d.code", "SELECT count(*) FROM emp e LEFT JOIN dept d ON e.dept = d.code", "SELECT d.code, p.title FROM dept d JOIN proj p ON 1 = 1 WHERE p.pid < 2",
 "SELECT e.dept, count(p.pid) FROM emp e LEFT JOIN proj p ON p.owner = e.id GROUP BY e.dept", "SELECT e.name, b.name, bb.name FROM emp e JOIN emp b ON e.boss = b.id JOIN emp bb ON b.boss = bb.id",
 "SELECT e.name FROM emp e JOIN proj p ON p.owner = e.id WHERE p.hours > 20 ORDER BY e.name", "SELECT d.title, e.name FROM dept d LEFT JOIN emp e ON e.dept = d.code AND e.salary > 100",
 "SELECT e.id, d.code FROM emp e JOIN dept d ON e.dept = d.code ORDER BY d.code DESC, e.id", "SELECT t1.id FROM emp t1 JOIN emp t2 ON t1.id = t2.boss GROUP BY t1.id",
 "SELECT emp.name, dept.title FROM emp JOIN dept ON emp.dept = dept.code WHERE emp.id < 5", "SELECT e.name, coalesce(d.title, 'n/a') FROM emp e LEFT JOIN dept d ON d.code = e.dept ORDER BY e.id",
])

# ---- s12 subqueries & compound
Q("s12", [
 "SELECT name FROM emp WHERE dept IN (SELECT code FROM dept WHERE budget > 400)", "SELECT code FROM dept WHERE code NOT IN (SELECT dept FROM emp)",
 "SELECT code FROM dept WHERE code NOT IN (SELECT dept FROM emp WHERE dept IS NOT NULL)", "SELECT name FROM emp WHERE salary = (SELECT max(salary) FROM emp)",
 "SELECT name, (SELECT title FROM dept WHERE code = 'eng') FROM emp WHERE id < 3", "SELECT (SELECT count(*) FROM emp)", "SELECT (SELECT id FROM emp WHERE id > 1000)",
 "SELECT 1 WHERE EXISTS (SELECT 1 FROM emp WHERE salary > 1000)", "SELECT 1 WHERE NOT EXISTS (SELECT 1 FROM emp WHERE salary > 1000)", "SELECT 1 WHERE EXISTS (SELECT 1 FROM emp)",
 "SELECT dept FROM emp UNION SELECT code FROM dept", "SELECT dept FROM emp UNION ALL SELECT code FROM dept", "SELECT id FROM emp WHERE id<3 UNION ALL SELECT pid FROM proj WHERE pid<3 ORDER BY 1 DESC",
 "SELECT id AS k FROM emp WHERE id < 3 UNION SELECT pid FROM proj WHERE pid < 5 ORDER BY k", "SELECT city FROM emp UNION SELECT title FROM dept ORDER BY 1 LIMIT 4",
 "SELECT avg(n) FROM (SELECT count(*) n FROM emp GROUP BY dept)", "SELECT name FROM emp WHERE salary > (SELECT avg(salary) FROM emp)", "SELECT name FROM emp WHERE id IN (SELECT boss FROM emp)",
 "SELECT name FROM emp WHERE id NOT IN (SELECT boss FROM emp WHERE boss IS NOT NULL)", "SELECT id FROM emp WHERE salary IN (SELECT salary FROM emp WHERE dept = 'sales')",
 "SELECT (SELECT max(hours) FROM proj) - (SELECT min(hours) FROM proj)", "SELECT id, (SELECT title FROM proj WHERE owner = 2 ORDER BY pid LIMIT 1) FROM emp WHERE id < 3",
 "SELECT * FROM (SELECT id, salary FROM emp WHERE salary > 90) ORDER BY id", "SELECT s.m FROM (SELECT max(salary) m FROM emp) s", "SELECT dept FROM emp WHERE dept IS NOT NULL UNION SELECT 'zzz' ORDER BY 1 DESC",
 "SELECT 1 UNION SELECT 1 UNION SELECT 2", "SELECT 1 UNION ALL SELECT 1 UNION ALL SELECT 2", "SELECT salary FROM emp UNION SELECT hours FROM proj",
 "SELECT id, name FROM emp WHERE id = (SELECT min(id) FROM emp WHERE dept = 'ops')", "SELECT name FROM emp WHERE 3 IN (SELECT pid FROM proj WHERE hours > 100)",
])
mk("s12", ["CREATE TABLE copy(a INTEGER, b TEXT)", "INSERT INTO copy SELECT id, name FROM emp WHERE id < 4", "INSERT INTO copy(b, a) SELECT dept, id * 10 FROM emp WHERE id IN (4,5)", "SELECT * FROM copy ORDER BY a"])
mk("s12", ["CREATE TABLE tot(d TEXT, s INTEGER)", "INSERT INTO tot SELECT dept, sum(salary) FROM emp GROUP BY dept", "SELECT * FROM tot ORDER BY d"])

# ---- s13 correlated
Q("s13", [
 "SELECT name FROM emp e WHERE salary > (SELECT avg(salary) FROM emp WHERE dept = e.dept)", "SELECT code FROM dept d WHERE EXISTS (SELECT 1 FROM emp e WHERE e.dept = d.code)",
 "SELECT code FROM dept d WHERE NOT EXISTS (SELECT 1 FROM emp e WHERE e.dept = d.code)", "SELECT name, (SELECT count(*) FROM emp s WHERE s.boss = e.id) FROM emp e",
 "SELECT name FROM emp e WHERE salary = (SELECT max(salary) FROM emp WHERE dept = e.dept)", "SELECT title FROM proj p WHERE hours > (SELECT avg(hours) FROM proj WHERE owner = p.owner)",
 "SELECT e.name FROM emp e WHERE e.id IN (SELECT owner FROM proj WHERE proj.hours > 20)", "SELECT d.code, (SELECT max(salary) FROM emp WHERE emp.dept = d.code) m FROM dept d ORDER BY d.code",
 "SELECT name FROM emp e WHERE (SELECT count(*) FROM proj WHERE owner = e.id) >= 2", "SELECT name FROM emp e WHERE EXISTS (SELECT 1 FROM proj p WHERE p.owner = e.id AND p.hours > 30)",
 "SELECT name FROM emp e WHERE NOT EXISTS (SELECT 1 FROM emp s WHERE s.boss = e.id)", "SELECT e.name, (SELECT d.title FROM dept d WHERE d.code = e.dept) FROM emp e",
 "SELECT e.id FROM emp e WHERE e.salary >= ALL_PLACEHOLDER" if False else "SELECT e.id FROM emp e WHERE (SELECT count(*) FROM emp x WHERE x.salary > e.salary) < 2",
 "SELECT d.code FROM dept d WHERE (SELECT sum(salary) FROM emp WHERE dept = d.code) > 200", "SELECT e.name FROM emp e WHERE e.salary > (SELECT avg(x.salary) FROM emp x WHERE x.city = e.city)",
 "SELECT name FROM emp e WHERE EXISTS (SELECT 1 FROM dept d WHERE d.code = e.dept AND d.budget > (SELECT avg(budget) FROM dept))",
 "SELECT p.title, (SELECT name FROM emp WHERE id = p.owner) FROM proj p", "SELECT e.name FROM emp e WHERE e.boss IN (SELECT id FROM emp b WHERE b.dept = e.dept)",
 "SELECT dept, (SELECT count(*) FROM emp y WHERE y.dept = x.dept) c FROM emp x GROUP BY dept", "SELECT e.id, (SELECT count(*) FROM proj p WHERE p.owner = e.id) np FROM emp e ORDER BY np DESC, e.id LIMIT 3",
])
for s in ["UPDATE emp SET salary = (SELECT budget/10 FROM dept WHERE code = emp.dept) WHERE dept='hr'", "DELETE FROM emp WHERE NOT EXISTS (SELECT 1 FROM proj WHERE owner = emp.id)",
          "UPDATE proj SET hours = (SELECT salary FROM emp WHERE emp.id = proj.owner)", "DELETE FROM dept WHERE NOT EXISTS (SELECT 1 FROM emp WHERE emp.dept = dept.code)",
          "UPDATE emp SET rating = (SELECT avg(rating) FROM emp x WHERE x.dept = emp.dept) WHERE rating IS NULL"]:
    tbl = re.search(r"(?:UPDATE|FROM)\s+(\w+)", s).group(1)
    mk("s13", [s, f"SELECT * FROM {tbl} ORDER BY 1"])

# ---- s14 persistence
def pcase(phases, preload=None, dump=("emp",), torn=None):
    """phases: list of step lists. oracle conn persists across phases, open txn lost at phase end."""
    o = Oracle()
    out = []
    for ph in phases:
        steps = []
        for s in ph:
            if isinstance(s, str):
                s = (s, ())
            steps.append({"sql": s[0], "params": list(s[1]), "expect": o.step(s[0], s[1])})
        o.close()
        out.append({"steps": steps})
    dump_t = {}
    for t in dump:
        try:
            dump_t[t] = [list(r) for r in o.con.execute(f"SELECT * FROM {t} ORDER BY rowid")]
        except sqlite3.Error:
            pass
    c = {"id": f"s14-{len([x for x in cases if x['ms']=='s14']):03d}", "ms": "s14", "kind": "persist", "phases": out,
         "dump": {"tables": dump_t} if dump_t else None}
    if preload is not None:
        c["files"] = {"db.log": preload}
    cases.append(c)


def logtext(txns, torn=None):
    t = "MINISQL-LOG 1\n" + "".join(json.dumps({"txn": [[s, list(p)] for s, p in tx]}, separators=(",", ":")) + "\n" for tx in txns)
    return t + (torn or "")


T = "CREATE TABLE emp(id INTEGER PRIMARY KEY, name TEXT NOT NULL, dept TEXT, salary INTEGER)"
pcase([[T, "INSERT INTO emp VALUES (1,'a','x',10),(2,'b','y',20)"], ["SELECT * FROM emp", "INSERT INTO emp(name) VALUES ('c')"], ["SELECT id, name FROM emp ORDER BY id"]])
pcase([[T], ["INSERT INTO emp VALUES (1,'a','x',10)"], ["UPDATE emp SET salary = salary * 2"], ["DELETE FROM emp WHERE id = 2", "INSERT INTO emp VALUES (3,'c','z',5)"], ["SELECT * FROM emp ORDER BY id"]])
pcase([[T, "INSERT INTO emp VALUES (1,'a','x',10)"], ["BEGIN", "INSERT INTO emp VALUES (2,'b','y',20)", "ROLLBACK", "SELECT count(*) FROM emp"], ["SELECT count(*) FROM emp"]])
pcase([[T, "INSERT INTO emp VALUES (1,'a','x',10)"], ["BEGIN", "INSERT INTO emp VALUES (2,'b','y',20)", "SELECT count(*) FROM emp"], ["SELECT count(*) FROM emp"]])
pcase([[T, "BEGIN", "INSERT INTO emp VALUES (1,'a','x',10)", "UPDATE emp SET name='A' WHERE id=1", "INSERT INTO emp VALUES (2,'b','y',20)", "COMMIT"], ["SELECT * FROM emp ORDER BY id"]])
pcase([[T, "INSERT INTO emp VALUES (1,'a','x',10)", "INSERT INTO emp VALUES (1,'dup','x',1)", "INSERT INTO emp(id) VALUES (9)", "INSERT INTO emp VALUES (2,'b','y',20)"], ["SELECT * FROM emp ORDER BY id"]])
pcase([[T, ("INSERT INTO emp VALUES (?,?,?,?)", (1, "it's \"q\" \\ \n tab\t", "déjà", 5)), ("INSERT INTO emp VALUES (?,?,?,?)", (2, "日本", None, None))], ["SELECT * FROM emp ORDER BY id"]])
pcase([["CREATE TABLE r(a REAL, b INTEGER)", ("INSERT INTO r VALUES (?,?)", (1.5, 2)), ("INSERT INTO r VALUES (?,?)", (3.0, None))], ["SELECT a, b, typeof(a) FROM r ORDER BY a"]], dump=("r",))
pcase([[T, "INSERT INTO emp VALUES (1,'a','x',10)", "DROP TABLE emp"], ["SELECT * FROM emp", "CREATE TABLE emp(z INTEGER)", "INSERT INTO emp VALUES (4)"], ["SELECT * FROM emp"]])
pcase([[T, "INSERT INTO emp(name) VALUES ('a'),('b')", "DELETE FROM emp"], ["INSERT INTO emp(name) VALUES ('c')", "SELECT id FROM emp"]])
pcase([[T, "CREATE TABLE other(x INTEGER)", "INSERT INTO other VALUES (1)", "INSERT INTO emp VALUES (1,'a','x',1)"], ["UPDATE other SET x = x + 1", "SELECT x FROM other"], ["SELECT x FROM other"]], dump=("emp", "other"))
# preloaded files (written as the reference writes them)
good = [[(T, [])], [("INSERT INTO emp VALUES (?,?,?,?)", [1, "a", "x", 10]), ("INSERT INTO emp VALUES (2,'b','y',20)", [])], [("UPDATE emp SET salary = salary + ?", [1])]]
pcase([["SELECT * FROM emp ORDER BY id", "INSERT INTO emp VALUES (3,'c','z',30)"], ["SELECT * FROM emp ORDER BY id"]], preload=logtext(good), dump=("emp",))
# the oracle for preload cases must start from the preloaded state: patch expectations by replaying the preload into the oracle first
def pcase_pre(txns, phases, torn=None, dump=("emp",)):
    o = Oracle()
    for tx in txns:
        for s, p in tx:
            o.con.execute(s, p)
    out = []
    for ph in phases:
        steps = []
        for s in ph:
            if isinstance(s, str):
                s = (s, ())
            steps.append({"sql": s[0], "params": list(s[1]), "expect": o.step(s[0], s[1])})
        o.close()
        out.append({"steps": steps})
    dt = {t: [list(r) for r in o.con.execute(f"SELECT * FROM {t} ORDER BY rowid")] for t in dump}
    cases.append({"id": f"s14-{len([x for x in cases if x['ms']=='s14']):03d}", "ms": "s14", "kind": "persist", "phases": out,
                  "files": {"db.log": logtext(txns, torn)}, "dump": {"tables": dt}})


cases.pop()   # drop the first (oracle-from-empty) preload attempt; replaced by pcase_pre below
pcase_pre(good, [["SELECT * FROM emp ORDER BY id", "INSERT INTO emp VALUES (3,'c','z',30)"], ["SELECT * FROM emp ORDER BY id"]])
pcase_pre(good, [["SELECT count(*) FROM emp", "UPDATE emp SET name = upper(name)"], ["SELECT name FROM emp ORDER BY id"]], torn='{"txn":[["INSERT INTO emp VALUES (9,\'t\',\'t\',9)",[]')
pcase_pre(good, [["SELECT count(*) FROM emp", "INSERT INTO emp VALUES (5,'n','m',1)"], ["SELECT id FROM emp ORDER BY id"]], torn='{"txn":[["INSERT INTO emp VALUES (9,\'t\',\'t\',9)",[]]]}')
pcase_pre(good[:1], [["SELECT count(*) FROM emp", "INSERT INTO emp(name) VALUES ('first')"], ["SELECT * FROM emp"]], torn='{"tx')
pcase_pre([[("CREATE TABLE emp(id INTEGER PRIMARY KEY, name TEXT NOT NULL, dept TEXT, salary INTEGER)", [])], [("INSERT INTO emp(name) VALUES ('a')", [])], [("INSERT INTO emp(name) VALUES ('b')", [])]],
          [["INSERT INTO emp(name) VALUES ('c')", "SELECT id, name FROM emp ORDER BY id"]])

# ---- s15 CLI
def clicase(runs, files=None):
    o = Oracle()
    out = []
    for r in runs:
        s = ""
        for st in minisql.split_statements(r["script"]):
            e = o.step(st)
            if "err" in e:
                s += f"ERROR {e['err']}\n"
            elif "rows" in e:
                for row in e["rows"]:
                    s += "|".join(minisql.fmt(v) for v in row) + "\n"
        o.close()
        out.append({"db": r["db"], "script": r["script"], "stdout": s})
    cases.append({"id": f"s15-{len([x for x in cases if x['ms']=='s15']):03d}", "ms": "s15", "kind": "cli", "runs": out, "files": files or {}})


FIXSCRIPT = ";\n".join(FIX) + ";\n"
for q in ["SELECT id, name, dept, salary, rating FROM emp ORDER BY id;", "SELECT * FROM dept ORDER BY code;", "SELECT 1, 2.5, 'x', NULL, 1.0, 100.0, 0.1+0.2;",
          "SELECT dept, count(*), avg(salary) FROM emp GROUP BY dept ORDER BY dept;", "SELECT nope FROM emp; SELECT 1;", "SELECT 'a;b', \"id\" FROM emp WHERE id=1;",
          "INSERT INTO dept VALUES ('z','Z',1);\nSELECT code FROM dept WHERE code='z';", "SELEC 1;\nSELECT 2;", "SELECT name FROM emp WHERE id=1\n;\n\n;SELECT 3",
          "BEGIN; INSERT INTO dept VALUES ('q','Q',1); ROLLBACK; SELECT count(*) FROM dept;", "COMMIT; SELECT 'after';", "SELECT 1e20, 1.5e-7, 123456789.125, -0.0, 3.0*2;",
          "SELECT '', 'multi\nline'", "SELECT id FROM emp WHERE id < 0;", "SELECT avg(rating), sum(rating), min(rating) FROM emp;"]:
    clicase([{"db": ":memory:", "script": FIXSCRIPT + q}])
clicase([{"db": "db.log", "script": "CREATE TABLE t(a INTEGER PRIMARY KEY, b TEXT);\nINSERT INTO t(b) VALUES ('one'),('two');\n"},
         {"db": "db.log", "script": "SELECT * FROM t;\nINSERT INTO t(b) VALUES ('three');\n"},
         {"db": "db.log", "script": "SELECT a, b FROM t ORDER BY a;\nSELECT bad;\n"}])
clicase([{"db": "db.log", "script": "CREATE TABLE t(a INTEGER);\nBEGIN;\nINSERT INTO t VALUES (1);\n"}, {"db": "db.log", "script": "SELECT count(*) FROM t;"}])

# ---- split: public sample (~10 percent), rest hidden
R = random.Random(14)
by = {}
for c in cases:
    by.setdefault(c["ms"], []).append(c)
public, hidden = [], []
for ms, cs in sorted(by.items()):
    if len(cs) > 24:                      # cap per milestone, seeded, keep original order
        keep = set(R.sample(range(len(cs)), 24))
        cs = [c for i, c in enumerate(cs) if i in keep]
    k = max(1, round(len(cs) * 0.1))
    pick = set(R.sample(range(len(cs)), k))
    for i, c in enumerate(cs):
        (public if i in pick else hidden).append(c)
os.makedirs(PUB, exist_ok=True)
json.dump(hidden, open(os.path.join(HID, "cases.json"), "w"))
json.dump(public, open(os.path.join(PUB, "public_cases.json"), "w"), indent=0)
print("hidden", len(hidden), "public", len(public), {m: len(v) for m, v in sorted(by.items())})
