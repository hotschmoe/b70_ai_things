"""minisql reference implementation (hidden from the agent)."""
import copy, json, math, os, re, sys
from decimal import Decimal, ROUND_HALF_UP


class SQLError(Exception):
    def __init__(self, code, msg=""):
        super().__init__(f"{code}: {msg}")
        self.code, self.msg = code, msg


class Result:
    def __init__(self, rows=None, columns=None, rowcount=-1):
        self.rows = rows if rows is not None else []
        self.columns = columns if columns is not None else []
        self.rowcount = rowcount


RESERVED = set("""select from where group by having order limit offset distinct as and or not in is null like between
exists case when then else end join inner left outer on union all asc desc create table if insert into values update set
delete drop begin commit rollback primary default unique""".split())
AGG = {"count", "sum", "avg", "min", "max"}


# ------------------------------------------------------------------ lexer
def tokenize(s):
    toks, i, n = [], 0, len(s)
    while i < n:
        c = s[i]
        if c.isspace():
            i += 1
        elif c.isdigit() or (c == "." and i + 1 < n and s[i + 1].isdigit()):
            m = re.compile(r"\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?").match(s, i)
            t = m.group()
            toks.append(("num", float(t) if re.search(r"[.eE]", t) else int(t), i, m.end()))
            i = m.end()
        elif c == "'":
            j, buf = i + 1, []
            while True:
                if j >= n:
                    raise SQLError("syntax", "unterminated string")
                if s[j] == "'":
                    if j + 1 < n and s[j + 1] == "'":
                        buf.append("'"); j += 2; continue
                    break
                buf.append(s[j]); j += 1
            toks.append(("str", "".join(buf), i, j + 1)); i = j + 1
        elif c == '"':
            j = s.find('"', i + 1)
            if j < 0:
                raise SQLError("syntax", "unterminated identifier")
            toks.append(("qid", s[i + 1:j], i, j + 1)); i = j + 1
        elif c.isalpha() or c == "_":
            m = re.compile(r"[A-Za-z_][A-Za-z0-9_]*").match(s, i)
            toks.append(("id", m.group(), i, m.end())); i = m.end()
        elif s.startswith(("<=", ">=", "<>", "!=", "||", "=="), i):
            toks.append(("op", s[i:i + 2], i, i + 2)); i += 2
        elif c in "+-*/%()<>=,.?;":
            toks.append(("op", c, i, i + 1)); i += 1
        else:
            raise SQLError("syntax", f"bad character {c!r}")
    toks.append(("eof", None, n, n))
    return toks


# ------------------------------------------------------------------ parser
class Parser:
    def __init__(self, sql):
        self.sql, self.t, self.p, self.nparam = sql, tokenize(sql), 0, 0

    def peek(self):
        return self.t[self.p]

    def kw(self, *names):
        k, v = self.t[self.p][0], self.t[self.p][1]
        return k == "id" and v.lower() in names

    def op(self, *ops):
        k, v = self.t[self.p][0], self.t[self.p][1]
        return k == "op" and v in ops

    def next(self):
        tok = self.t[self.p]; self.p += 1; return tok

    def accept_kw(self, name):
        if self.kw(name):
            self.p += 1; return True
        return False

    def accept_op(self, o):
        if self.op(o):
            self.p += 1; return True
        return False

    def expect_kw(self, name):
        if not self.accept_kw(name):
            raise SQLError("syntax", f"expected {name}")

    def expect_op(self, o):
        if not self.accept_op(o):
            raise SQLError("syntax", f"expected {o}")

    def ident(self):
        k, v = self.t[self.p][0], self.t[self.p][1]
        if k == "qid" or (k == "id" and v.lower() not in RESERVED):
            self.p += 1; return v
        raise SQLError("syntax", "expected identifier")

    # statements
    def statement(self):
        if self.kw("select"):
            node = self.select_full()
        elif self.kw("create"):
            node = self.create()
        elif self.kw("insert"):
            node = self.insert()
        elif self.kw("update"):
            node = self.update()
        elif self.kw("delete"):
            node = self.delete()
        elif self.kw("drop"):
            self.p += 1; self.expect_kw("table")
            ie = False
            if self.accept_kw("if"):
                self.expect_kw("exists"); ie = True
            node = ("drop", self.ident(), ie)
        elif self.kw("begin"):
            self.p += 1; node = ("begin",)
        elif self.kw("commit"):
            self.p += 1; node = ("commit",)
        elif self.kw("rollback"):
            self.p += 1; node = ("rollback",)
        else:
            raise SQLError("syntax", "unknown statement")
        self.accept_op(";")
        if self.peek()[0] != "eof":
            raise SQLError("syntax", "trailing input")
        return node

    def create(self):
        self.p += 1; self.expect_kw("table")
        ine = False
        if self.accept_kw("if"):
            self.expect_kw("not"); self.expect_kw("exists"); ine = True
        name = self.ident(); self.expect_op("(")
        cols = []
        while True:
            c = {"name": self.ident(), "type": None, "pk": False, "notnull": False, "unique": False, "default": None}
            if self.kw("integer", "text", "real"):
                c["type"] = self.next()[1].upper()
            else:
                raise SQLError("syntax", "column type required")
            while True:
                if self.accept_kw("primary"):
                    if not self.accept_kw("key"):
                        raise SQLError("syntax", "expected KEY")
                    c["pk"] = True
                elif self.accept_kw("not"):
                    self.expect_kw("null"); c["notnull"] = True
                elif self.accept_kw("unique"):
                    c["unique"] = True
                elif self.accept_kw("default"):
                    neg = self.accept_op("-")
                    k, v = self.next()[:2]
                    if k not in ("num", "str") and not (k == "id" and v.lower() == "null"):
                        raise SQLError("syntax", "bad default")
                    v = None if k == "id" else v
                    if neg:
                        v = -v
                    c["default"] = v
                else:
                    break
            cols.append(c)
            if not self.accept_op(","):
                break
        self.expect_op(")")
        if sum(1 for c in cols if c["pk"]) > 1:
            raise SQLError("syntax", "multiple primary keys")
        return ("create", name, cols, ine)

    def insert(self):
        self.p += 1; self.expect_kw("into")
        name = self.ident(); cols = None
        if self.accept_op("("):
            cols = [self.ident()]
            while self.accept_op(","):
                cols.append(self.ident())
            self.expect_op(")")
        if self.accept_kw("values"):
            rows = []
            while True:
                self.expect_op("(")
                r = [self.expr()]
                while self.accept_op(","):
                    r.append(self.expr())
                self.expect_op(")"); rows.append(r)
                if not self.accept_op(","):
                    break
            return ("insert", name, cols, ("values", rows))
        if self.kw("select"):
            return ("insert", name, cols, ("select", self.select_full()))
        raise SQLError("syntax", "expected VALUES or SELECT")

    def update(self):
        self.p += 1; name = self.ident(); self.expect_kw("set")
        sets = []
        while True:
            c = self.ident(); self.expect_op("=")
            sets.append((c, self.expr()))
            if not self.accept_op(","):
                break
        w = self.expr() if self.accept_kw("where") else None
        return ("update", name, sets, w)

    def delete(self):
        self.p += 1; self.expect_kw("from"); name = self.ident()
        w = self.expr() if self.accept_kw("where") else None
        return ("delete", name, w)

    def select_full(self):
        cores, alls = [self.select_core()], []
        while self.accept_kw("union"):
            alls.append(self.accept_kw("all"))
            if not self.kw("select"):
                raise SQLError("syntax", "expected SELECT")
            cores.append(self.select_core())
        order = []
        if self.accept_kw("order"):
            self.expect_kw("by")
            while True:
                e = self.expr(); d = False
                if self.accept_kw("desc"):
                    d = True
                else:
                    self.accept_kw("asc")
                order.append((e, d))
                if not self.accept_op(","):
                    break
        lim = off = None
        if self.accept_kw("limit"):
            lim = self.expr()
            if self.accept_kw("offset"):
                off = self.expr()
        return {"cores": cores, "alls": alls, "order": order, "limit": lim, "offset": off}

    def select_core(self):
        self.expect_kw("select")
        distinct = self.accept_kw("distinct")
        if not distinct:
            self.accept_kw("all")
        items = []
        while True:
            if self.accept_op("*"):
                items.append(("*", None, None))
            elif (self.peek()[0] in ("id", "qid") and self.t[self.p + 1][1] == "." and self.t[self.p + 2][1] == "*"
                  and self.t[self.p + 1][0] == "op"):
                t = self.ident(); self.p += 2
                items.append(("t*", t, None))
            else:
                start = self.peek()[2]
                e = self.expr()
                end = self.t[self.p - 1][3]
                alias = None
                if self.accept_kw("as"):
                    alias = self.ident()
                elif self.peek()[0] == "qid" or (self.peek()[0] == "id" and self.peek()[1].lower() not in RESERVED):
                    alias = self.ident()
                items.append(("e", e, alias or (e[2] if e[0] == "col" else self.sql[start:end])))
            if not self.accept_op(","):
                break
        frm, joins = None, []
        if self.accept_kw("from"):
            frm = self.source()
            while True:
                kind = None
                if self.accept_kw("join"):
                    kind = "inner"
                elif self.kw("inner"):
                    self.p += 1; self.expect_kw("join"); kind = "inner"
                elif self.kw("left"):
                    self.p += 1; self.accept_kw("outer"); self.expect_kw("join"); kind = "left"
                if kind is None:
                    break
                src = self.source(); self.expect_kw("on")
                joins.append((kind, src, self.expr()))
        where = self.expr() if self.accept_kw("where") else None
        group, having = [], None
        if self.accept_kw("group"):
            self.expect_kw("by")
            group.append(self.expr())
            while self.accept_op(","):
                group.append(self.expr())
        if self.accept_kw("having"):
            having = self.expr()
        return {"distinct": distinct, "items": items, "from": frm, "joins": joins, "where": where, "group": group, "having": having}

    def source(self):
        if self.accept_op("("):
            sel = self.select_full(); self.expect_op(")")
            alias = ""
            if self.accept_kw("as") or self.peek()[0] == "qid" or (self.peek()[0] == "id" and self.peek()[1].lower() not in RESERVED):
                alias = self.ident()
            return ("sub", sel, alias)
        name = self.ident(); alias = name
        if self.accept_kw("as"):
            alias = self.ident()
        elif self.peek()[0] == "qid" or (self.peek()[0] == "id" and self.peek()[1].lower() not in RESERVED):
            alias = self.ident()
        return ("table", name, alias)

    # expressions
    def expr(self):
        l = self.and_expr()
        while self.accept_kw("or"):
            l = ("or", l, self.and_expr())
        return l

    def and_expr(self):
        l = self.not_expr()
        while self.accept_kw("and"):
            l = ("and", l, self.not_expr())
        return l

    def not_expr(self):
        if self.accept_kw("not"):
            return ("not", self.not_expr())
        return self.eq_expr()

    def eq_expr(self):
        l = self.rel_expr()
        while True:
            if self.op("=", "==", "<>", "!="):
                o = self.next()[1]
                l = ("bin", {"==": "=", "!=": "<>"}.get(o, o), l, self.rel_expr())
            elif self.kw("is"):
                self.p += 1
                neg = self.accept_kw("not")
                l = ("is", l, self.rel_expr(), neg)
            else:
                neg = False
                save = self.p
                if self.kw("not"):
                    self.p += 1; neg = True
                if self.accept_kw("in"):
                    self.expect_op("(")
                    if self.kw("select"):
                        sel = self.select_full(); self.expect_op(")")
                        l = ("in", l, ("sel", sel), neg)
                    elif self.accept_op(")"):
                        l = ("in", l, ("list", []), neg)
                    else:
                        items = [self.expr()]
                        while self.accept_op(","):
                            items.append(self.expr())
                        self.expect_op(")")
                        l = ("in", l, ("list", items), neg)
                elif self.accept_kw("like"):
                    l = ("like", l, self.rel_expr(), neg)
                elif self.accept_kw("between"):
                    lo = self.rel_expr(); self.expect_kw("and"); hi = self.rel_expr()
                    l = ("between", l, lo, hi, neg)
                else:
                    self.p = save
                    return l

    def rel_expr(self):
        l = self.add_expr()
        while self.op("<", "<=", ">", ">="):
            o = self.next()[1]
            l = ("bin", o, l, self.add_expr())
        return l

    def add_expr(self):
        l = self.mul_expr()
        while self.op("+", "-"):
            o = self.next()[1]
            l = ("bin", o, l, self.mul_expr())
        return l

    def mul_expr(self):
        l = self.cat_expr()
        while self.op("*", "/", "%"):
            o = self.next()[1]
            l = ("bin", o, l, self.cat_expr())
        return l

    def cat_expr(self):
        l = self.unary()
        while self.op("||"):
            self.p += 1
            l = ("bin", "||", l, self.unary())
        return l

    def unary(self):
        if self.accept_op("-"):
            return ("neg", self.unary())
        if self.accept_op("+"):
            return self.unary()
        return self.primary()

    def primary(self):
        k, v = self.peek()[0], self.peek()[1]
        if k in ("num", "str"):
            self.p += 1; return ("lit", v)
        if self.accept_op("?"):
            self.nparam += 1; return ("param", self.nparam - 1)
        if self.accept_op("("):
            if self.kw("select"):
                sel = self.select_full(); self.expect_op(")"); return ("subq", sel)
            e = self.expr(); self.expect_op(")"); return e
        if k == "id" and v.lower() == "null":
            self.p += 1; return ("lit", None)
        if k == "id" and v.lower() == "exists":
            self.p += 1; self.expect_op("(")
            sel = self.select_full(); self.expect_op(")"); return ("exists", sel)
        if k == "id" and v.lower() == "case":
            self.p += 1
            operand = None if self.kw("when") else self.expr()
            whens = []
            while self.accept_kw("when"):
                c = self.expr(); self.expect_kw("then"); whens.append((c, self.expr()))
            if not whens:
                raise SQLError("syntax", "CASE needs WHEN")
            els = self.expr() if self.accept_kw("else") else None
            self.expect_kw("end")
            return ("case", operand, whens, els)
        if k in ("id", "qid") and not (k == "id" and v.lower() in RESERVED):
            self.p += 1
            if k == "id" and self.op("("):
                self.p += 1
                name = v.lower(); distinct = self.accept_kw("distinct"); args = []; star = False
                if self.accept_op("*"):
                    star = True
                elif not self.op(")"):
                    args.append(self.expr())
                    while self.accept_op(","):
                        args.append(self.expr())
                self.expect_op(")")
                return ("func", name, args, distinct, star)
            if self.op(".") and self.t[self.p + 1][0] in ("id", "qid"):
                self.p += 1
                return ("col", v, self.ident())
            return ("col", None, v)
        raise SQLError("syntax", f"unexpected token {v!r}")


# ------------------------------------------------------------------ values
def cls(v):
    return -1 if v is None else (0 if isinstance(v, (int, float)) else 1)


def sk(v):
    return (cls(v), 0 if v is None else v)


def cmp(a, b):
    ka, kb = sk(a), sk(b)
    return (ka > kb) - (ka < kb)


NUMRE = re.compile(r"\s*[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?")


def tonum(v):
    if isinstance(v, (int, float)):
        return v
    m = NUMRE.match(v)
    if not m:
        return 0
    t = m.group().strip()
    return float(t) if re.search(r"[.eE]", t) else int(t)


def totext(v):
    if isinstance(v, str):
        return v
    if isinstance(v, float):
        return repr(v)
    return str(v)


def alower(s):
    return s.translate({i: i + 32 for i in range(65, 91)})


def aupper(s):
    return s.translate({i: i - 32 for i in range(97, 123)})


def like(s, p):
    s, p = alower(totext(s)), alower(totext(p))
    rx = "".join(".*" if c == "%" else "." if c == "_" else re.escape(c) for c in p)
    return re.fullmatch(rx, s, re.S) is not None


def b3(v):
    return None if v is None else (1 if (v != 0 if not isinstance(v, str) else tonum(v) != 0) else 0)


def intdiv(a, b):
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b > 0) else -q


def coerce(t, v):
    if v is None:
        return None
    if t == "INTEGER":
        if isinstance(v, float) and v == int(v) and abs(v) < 2 ** 63:
            return int(v)
        if isinstance(v, str) and re.fullmatch(r"[+-]?\d+", v.strip() or "x"):
            return int(v)
    elif t == "REAL":
        if isinstance(v, int):
            return float(v)
        if isinstance(v, str) and NUMRE.fullmatch(v):
            return float(v)
    elif t == "TEXT":
        if isinstance(v, (int, float)):
            return totext(v)
    return v


class Env:
    def __init__(self, sources, outer=None):
        self.sources, self.outer = sources, outer   # sources: [(alias, cols, vals)]

    def lookup(self, table, name):
        ln = name.lower(); hits = []
        for al, cols, vals in self.sources:
            if table is None or al.lower() == table.lower():
                for i, c in enumerate(cols):
                    if c.lower() == ln:
                        hits.append(vals[i])
        if len(hits) > 1:
            raise SQLError("ambiguous", name)
        if hits:
            return hits[0]
        if self.outer is not None:
            return self.outer.lookup(table, name)
        raise SQLError("no_column", name)


def has_agg(e):
    if isinstance(e, tuple):
        if e and e[0] == "func" and (e[1] in AGG and (e[1] not in ("min", "max") or len(e[2]) == 1)):
            return True
        if e and e[0] in ("subq", "exists"):
            return False
        return any(has_agg(x) for x in e)
    if isinstance(e, list):
        return any(has_agg(x) for x in e)
    return False


class Table:
    def __init__(self, name, cols):
        self.name, self.cols, self.rows = name, cols, []
        self.ipk = next((i for i, c in enumerate(cols) if c["pk"] and c["type"] == "INTEGER"), None)

    def colnames(self):
        return [c["name"] for c in self.cols]


class Database:
    def __init__(self, path=None):
        self.tables = {}
        self.in_txn = False
        self.snap = None
        self.pending = []
        self.path = None if path in (None, ":memory:") else path
        self.fh = None
        if self.path:
            self._open_file()

    # --- persistence
    def _open_file(self):
        if not os.path.exists(self.path) or os.path.getsize(self.path) == 0:
            with open(self.path, "w") as f:
                f.write("MINISQL-LOG 1\n"); f.flush(); os.fsync(f.fileno())
        else:
            data = open(self.path, "rb").read().decode("utf-8")
            lines = data.split("\n")
            if lines[0] != "MINISQL-LOG 1":
                raise SQLError("corrupt", "bad header")
            good = len(lines[0]) + 1
            tail = lines[-1]            # text after last newline ('' if file ends with newline)
            body = lines[1:-1]
            for ln in body:
                try:
                    obj = json.loads(ln)
                except ValueError:
                    raise SQLError("corrupt", "bad log line")
                self._replay(obj)
                good += len(ln.encode("utf-8")) + 1
            if tail:
                # torn final line: ignored; a complete JSON line missing only its newline is also ignored
                with open(self.path, "r+b") as f:
                    f.truncate(len(data[:len(data) - len(tail)].encode("utf-8")))
        self.fh = open(self.path, "a", encoding="utf-8")

    def _replay(self, obj):
        for sql, params in obj["txn"]:
            self._run(sql, tuple(params), log=False)

    def _write_log(self, entries):
        if self.fh is None or not entries:
            return
        self.fh.write(json.dumps({"txn": entries}, separators=(",", ":")) + "\n")
        self.fh.flush(); os.fsync(self.fh.fileno())

    def close(self):
        if self.fh:
            self.fh.close(); self.fh = None

    # --- API
    def execute(self, sql, params=()):
        return self._run(sql, tuple(params), log=True)

    def _run(self, sql, params, log):
        p = Parser(sql)
        node = p.statement()
        if p.nparam != len(params):
            raise SQLError("syntax", "wrong number of parameters")
        kind = node[0] if isinstance(node, tuple) else "select"
        if kind == "begin":
            if self.in_txn:
                raise SQLError("txn", "already in transaction")
            self.in_txn, self.snap, self.pending = True, copy.deepcopy(self.tables), []
            return Result(rowcount=0)
        if kind in ("commit", "rollback"):
            if not self.in_txn:
                raise SQLError("txn", "no transaction")
            if kind == "commit":
                if log:
                    self._write_log(self.pending)
            else:
                self.tables = self.snap
            self.in_txn, self.snap, self.pending = False, None, []
            return Result(rowcount=0)
        if kind == "select":
            cols, rows = self.select(node, params, None)
            return Result([tuple(r) for r in rows], cols, -1)
        before = copy.deepcopy(self.tables)
        try:
            n = self.write(node, params)
        except Exception:
            self.tables = before
            raise
        if log:
            if self.in_txn:
                self.pending.append([sql, list(params)])
            else:
                self._write_log([[sql, list(params)]])
        return Result(rowcount=n)

    # --- DDL / DML
    def table(self, name):
        t = self.tables.get(name.lower())
        if t is None:
            raise SQLError("no_table", name)
        return t

    def write(self, node, params):
        k = node[0]
        if k == "create":
            _, name, cols, ine = node
            if name.lower() in self.tables:
                if ine:
                    return 0
                raise SQLError("exists", name)
            seen = set()
            for c in cols:
                if c["name"].lower() in seen:
                    raise SQLError("syntax", "duplicate column")
                seen.add(c["name"].lower())
            self.tables[name.lower()] = Table(name, cols)
            return 0
        if k == "drop":
            if node[1].lower() not in self.tables:
                if node[2]:
                    return 0
                raise SQLError("no_table", node[1])
            del self.tables[node[1].lower()]
            return 0
        if k == "insert":
            return self.insert(node, params)
        if k == "update":
            return self.update(node, params)
        return self.delete(node, params)

    def check_row(self, t, row, skip=None):
        for i, c in enumerate(t.cols):
            if row[i] is None and (c["notnull"] or c["pk"]):
                raise SQLError("constraint", f"NOT NULL {t.name}.{c['name']}")
            if (c["pk"] or c["unique"]) and row[i] is not None:
                for r in t.rows:
                    if r is not skip and cmp(r[i], row[i]) == 0:
                        raise SQLError("constraint", f"UNIQUE {t.name}.{c['name']}")

    def insert(self, node, params):
        _, name, cols, src = node
        t = self.table(name)
        names = [c.lower() for c in t.colnames()]
        idx = list(range(len(t.cols))) if cols is None else []
        if cols is not None:
            for c in cols:
                if c.lower() not in names:
                    raise SQLError("no_column", c)
                idx.append(names.index(c.lower()))
        if src[0] == "values":
            rows = [[self.eval(e, Env([]), params) for e in r] for r in src[1]]
        else:
            rows = [list(r) for r in self.select(src[1], params, None)[1]]
        for vals in rows:
            if len(vals) != len(idx):
                raise SQLError("syntax", "column count mismatch")
            row = [c["default"] for c in t.cols]
            for i, v in zip(idx, vals):
                row[i] = v
            row = [coerce(c["type"], v) for c, v in zip(t.cols, row)]
            if t.ipk is not None and row[t.ipk] is None:
                row[t.ipk] = max([r[t.ipk] for r in t.rows], default=0) + 1
            self.check_row(t, row)
            t.rows.append(row)
        return len(rows)

    def update(self, node, params):
        _, name, sets, where = node
        t = self.table(name)
        names = [c.lower() for c in t.colnames()]
        for c, _e in sets:
            if c.lower() not in names:
                raise SQLError("no_column", c)
        env_cols = t.colnames()
        hit = []
        for r in t.rows:
            env = Env([(t.name, env_cols, r)])
            if where is None or b3(self.eval(where, env, params)) == 1:
                hit.append((r, [self.eval(e, env, params) for _c, e in sets]))
        for r, vals in hit:
            new = list(r)
            for (c, _e), v in zip(sets, vals):
                i = names.index(c.lower())
                new[i] = coerce(t.cols[i]["type"], v)
            self.check_row(t, new, skip=r)
            r[:] = new
        return len(hit)

    def delete(self, node, params):
        _, name, where = node
        t = self.table(name)
        keep, n = [], 0
        for r in t.rows:
            env = Env([(t.name, t.colnames(), r)])
            if where is None or b3(self.eval(where, env, params)) == 1:
                n += 1
            else:
                keep.append(r)
        t.rows = keep
        return n

    # --- SELECT
    def source_rows(self, src, params, outer):
        if src[0] == "table":
            t = self.table(src[1])
            return src[2], t.colnames(), [list(r) for r in t.rows]
        cols, rows = self.select(src[1], params, outer)
        return src[2], cols, [list(r) for r in rows]

    def select(self, node, params, outer):
        cores = node["cores"]
        if len(cores) == 1:
            cols, units = self.core(cores[0], params, outer)
            return self.finish(node, cols, units, params, outer, single=True)
        res = None
        for i, c in enumerate(cores):
            cols, units = self.core(c, params, outer)
            rows = [u[2] for u in units]
            if res is None:
                res, first = rows, cols
            else:
                if len(cols) != len(first):
                    raise SQLError("syntax", "column count mismatch in UNION")
                res = res + rows
                if not node["alls"][i - 1]:
                    seen, out = set(), []
                    for r in res:
                        k = tuple(sk(v) for v in r)
                        if k not in seen:
                            seen.add(k); out.append(r)
                    res = out
        return self.finish(node, first, [(None, None, r) for r in res], params, outer, single=False)

    def finish(self, node, cols, units, params, outer, single):
        if node["order"]:
            keys = []
            for e, desc in node["order"]:
                keys.append((e, desc))
            for e, desc in reversed(keys):
                def kf(u, e=e):
                    if e[0] == "lit" and isinstance(e[1], int) and not isinstance(e[1], bool):
                        if not 1 <= e[1] <= len(cols):
                            raise SQLError("syntax", "ORDER BY position out of range")
                        return sk(u[2][e[1] - 1])
                    if e[0] == "col" and e[1] is None:
                        m = [i for i, c in enumerate(cols) if c.lower() == e[2].lower()]
                        if m:
                            return sk(u[2][m[0]])
                    if not single:
                        raise SQLError("no_column", "ORDER BY in compound select must use output columns")
                    return sk(self.eval(e, u[0], params, u[1]))
                units = sorted(units, key=kf, reverse=desc)
        rows = [u[2] for u in units]
        if node["limit"] is not None:
            lim = self.eval(node["limit"], Env([]), params)
            off = self.eval(node["offset"], Env([]), params) if node["offset"] is not None else 0
            if not isinstance(lim, int) or not isinstance(off, int):
                raise SQLError("syntax", "LIMIT must be integer")
            off = max(off, 0)
            rows = rows[off:] if lim < 0 else rows[off:off + lim]
        return cols, rows

    def core(self, c, params, outer):
        # build row combinations
        srcs = []        # (alias, cols)
        if c["from"] is None:
            combos = [[]]
        else:
            al, cols, rows = self.source_rows(c["from"], params, outer)
            srcs.append((al, cols))
            combos = [[r] for r in rows]
            for kind, src, on in c["joins"]:
                al2, cols2, rows2 = self.source_rows(src, params, outer)
                srcs2 = srcs + [(al2, cols2)]
                new = []
                for combo in combos:
                    m = False
                    for r2 in rows2:
                        env = Env([(a, cc, v) for (a, cc), v in zip(srcs2, combo + [r2])], outer)
                        if b3(self.eval(on, env, params)) == 1:
                            new.append(combo + [r2]); m = True
                    if not m and kind == "left":
                        new.append(combo + [[None] * len(cols2)])
                srcs, combos = srcs2, new
        envs = [Env([(a, cc, v) for (a, cc), v in zip(srcs, combo)], outer) for combo in combos]
        if c["where"] is not None:
            envs = [e for e in envs if b3(self.eval(c["where"], e, params)) == 1]
        # expand items
        items = []
        for kind, a, b in c["items"]:
            if kind == "e":
                items.append((a, b))
            else:
                if c["from"] is None:
                    raise SQLError("syntax", "no tables specified")
                matched = False
                for al, cc in srcs:
                    if kind == "*" or al.lower() == a.lower():
                        matched = True
                        items.extend((("col", al, n), n) for n in cc)
                if not matched:
                    raise SQLError("no_table", a)
        cols = [n for _e, n in items]
        agg = bool(c["group"]) or any(has_agg(e) for e, _n in items) or (c["having"] is not None and has_agg(c["having"]))
        if not agg:
            units = [(e, None, [self.eval(x, e, params) for x, _n in items]) for e in envs]
        else:
            groups = {}
            if c["group"]:
                for e in envs:
                    k = tuple(sk(self.eval(g, e, params)) for g in c["group"])
                    groups.setdefault(k, []).append(e)
                glist = list(groups.values())
            else:
                glist = [envs]
            units = []
            nullenv = Env([(a, cc, [None] * len(cc)) for a, cc in srcs], outer)
            for g in glist:
                rep = g[0] if g else nullenv
                if c["having"] is not None and b3(self.eval(c["having"], rep, params, g)) != 1:
                    continue
                units.append((rep, g, [self.eval(x, rep, params, g) for x, _n in items]))
        if c["distinct"]:
            seen, out = set(), []
            for u in units:
                k = tuple(sk(v) for v in u[2])
                if k not in seen:
                    seen.add(k); out.append(u)
            units = out
        return cols, units

    # --- expression evaluation
    def eval(self, e, env, params, group=None):
        k = e[0]
        ev = lambda x: self.eval(x, env, params, group)
        if k == "lit":
            return e[1]
        if k == "param":
            return params[e[1]]
        if k == "col":
            return env.lookup(e[1], e[2])
        if k == "neg":
            v = ev(e[1])
            return None if v is None else -tonum(v)
        if k == "not":
            v = b3(ev(e[1]))
            return None if v is None else 1 - v
        if k == "and":
            a = b3(ev(e[1]))
            if a == 0:
                return 0
            b = b3(ev(e[2]))
            if b == 0:
                return 0
            return None if a is None or b is None else 1
        if k == "or":
            a = b3(ev(e[1]))
            if a == 1:
                return 1
            b = b3(ev(e[2]))
            if b == 1:
                return 1
            return None if a is None or b is None else 0
        if k == "bin":
            return self.binop(e[1], ev(e[2]), ev(e[3]))
        if k == "is":
            a, b = ev(e[1]), ev(e[2])
            r = 1 if cmp(a, b) == 0 else 0
            return 1 - r if e[3] else r
        if k == "between":
            v, lo, hi = ev(e[1]), ev(e[2]), ev(e[3])
            r = self.eval(("and", ("lit", self.binop(">=", v, lo)), ("lit", self.binop("<=", v, hi))), env, params)
            return r if not e[4] or r is None else 1 - r
        if k == "like":
            a, p = ev(e[1]), ev(e[2])
            if a is None or p is None:
                return None
            r = 1 if like(a, p) else 0
            return 1 - r if e[3] else r
        if k == "in":
            v = ev(e[1])
            if e[2][0] == "list":
                vals = [ev(x) for x in e[2][1]]
            else:
                vals = [r[0] for r in self.select(e[2][1], params, env)[1]]
            if not vals:
                r = 0
            elif v is None:
                r = None
            elif any(x is not None and cmp(v, x) == 0 for x in vals):
                r = 1
            elif any(x is None for x in vals):
                r = None
            else:
                r = 0
            return r if not e[3] or r is None else 1 - r
        if k == "exists":
            return 1 if self.select(e[1], params, env)[1] else 0
        if k == "subq":
            cols, rows = self.select(e[1], params, env)
            if len(cols) != 1:
                raise SQLError("syntax", "scalar subquery must return one column")
            return rows[0][0] if rows else None
        if k == "case":
            if e[1] is None:
                for c, r in e[2]:
                    if b3(ev(c)) == 1:
                        return ev(r)
            else:
                v = ev(e[1])
                for c, r in e[2]:
                    w = ev(c)
                    if v is not None and w is not None and cmp(v, w) == 0:
                        return ev(r)
            return ev(e[3]) if e[3] is not None else None
        if k == "func":
            return self.func(e, env, params, group)
        raise SQLError("syntax", f"bad expression {k}")

    def binop(self, o, a, b):
        if o in ("=", "<>", "<", "<=", ">", ">="):
            if a is None or b is None:
                return None
            c = cmp(a, b)
            return int({"=": c == 0, "<>": c != 0, "<": c < 0, "<=": c <= 0, ">": c > 0, ">=": c >= 0}[o])
        if a is None or b is None:
            return None
        if o == "||":
            return totext(a) + totext(b)
        a, b = tonum(a), tonum(b)
        if o == "+": return a + b
        if o == "-": return a - b
        if o == "*": return a * b
        if o == "/":
            if b == 0:
                return None
            return intdiv(a, b) if isinstance(a, int) and isinstance(b, int) else a / b
        if o == "%":
            a, b = int(a), int(b)
            return None if b == 0 else a - b * intdiv(a, b)
        raise SQLError("syntax", o)

    def func(self, e, env, params, group):
        _, name, args, distinct, star = e
        if name in AGG and (name not in ("min", "max") or len(args) == 1):
            if group is None:
                raise SQLError("syntax", "misuse of aggregate")
            if star:
                if name != "count":
                    raise SQLError("syntax", "bad *")
                return len(group)
            if len(args) != 1:
                raise SQLError("syntax", "aggregate takes one argument")
            vals = [v for v in (self.eval(args[0], g, params) for g in group) if v is not None]
            if distinct:
                seen, out = set(), []
                for v in vals:
                    if sk(v) not in seen:
                        seen.add(sk(v)); out.append(v)
                vals = out
            if name == "count": return len(vals)
            if not vals: return None
            if name == "min": return min(vals, key=sk)
            if name == "max": return max(vals, key=sk)
            nums = [tonum(v) for v in vals]
            if name == "sum": return sum(nums)
            return sum(nums) / len(nums)
        a = [self.eval(x, env, params, group) for x in args]
        n = len(a)
        def need(*ks):
            if n not in ks:
                raise SQLError("syntax", f"wrong number of arguments to {name}")
        if name == "coalesce":
            if n < 2: need(2)
            return next((v for v in a if v is not None), None)
        if name == "ifnull":
            need(2); return a[0] if a[0] is not None else a[1]
        if name == "nullif":
            need(2); return None if a[0] is not None and a[1] is not None and cmp(a[0], a[1]) == 0 else a[0]
        if name == "typeof":
            need(1); v = a[0]
            return "null" if v is None else "integer" if isinstance(v, int) else "real" if isinstance(v, float) else "text"
        if name in ("min", "max"):
            if n < 2: need(2)
            if any(v is None for v in a): return None
            return (min if name == "min" else max)(a, key=sk)
        if name not in ("abs", "length", "lower", "upper", "substr", "round", "replace", "trim", "ltrim", "rtrim", "instr"):
            raise SQLError("syntax", f"no such function {name}")
        if any(v is None for v in a):
            return None
        if name == "abs": need(1); return abs(tonum(a[0]))
        if name == "length": need(1); return len(totext(a[0]))
        if name == "lower": need(1); return alower(totext(a[0]))
        if name == "upper": need(1); return aupper(totext(a[0]))
        if name == "substr":
            need(2, 3); s, st = totext(a[0]), int(tonum(a[1]))
            if st < 1 or (n == 3 and int(tonum(a[2])) < 0):
                raise SQLError("syntax", "substr arguments out of defined range")
            return s[st - 1:] if n == 2 else s[st - 1:st - 1 + int(tonum(a[2]))]
        if name == "round":
            need(1, 2); x = tonum(a[0]); d = int(tonum(a[1])) if n == 2 else 0
            q = Decimal(1).scaleb(-d)
            return float(Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP))
        if name == "replace":
            need(3); s, f, r = (totext(x) for x in a)
            return s if f == "" else s.replace(f, r)
        if name in ("trim", "ltrim", "rtrim"):
            need(1, 2); s = totext(a[0]); chars = totext(a[1]) if n == 2 else " "
            return {"trim": s.strip, "ltrim": s.lstrip, "rtrim": s.rstrip}[name](chars)
        if name == "instr":
            need(2); return totext(a[0]).find(totext(a[1])) + 1


# ------------------------------------------------------------------ CLI
def fmt(v):
    return "NULL" if v is None else repr(v) if isinstance(v, float) else str(v)


def split_statements(text):
    out, cur, q = [], [], None
    for ch in text:
        if q:
            cur.append(ch)
            if ch == q:
                q = None
        elif ch in ("'", '"'):
            q = ch; cur.append(ch)
        elif ch == ";":
            out.append("".join(cur)); cur = []
        else:
            cur.append(ch)
    out.append("".join(cur))
    return [s for s in out if s.strip()]


def main(argv):
    if len(argv) != 2:
        print("usage: minisql.py DBFILE", file=sys.stderr)
        return 2
    try:
        db = Database(argv[1])
    except SQLError as e:
        print(f"ERROR {e.code}")
        return 1
    for st in split_statements(sys.stdin.read()):
        try:
            r = db.execute(st)
            if r.rowcount == -1:
                for row in r.rows:
                    print("|".join(fmt(v) for v in row))
        except SQLError as e:
            print(f"ERROR {e.code}")
    db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
