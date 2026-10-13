// rx reference implementation (hidden from the agent). Backtracking VM over a compiled program.
use std::fs;
use std::io::Write;

type R<T> = Result<T, &'static str>;
const NONE: usize = usize::MAX;

#[derive(Clone, Copy)]
struct Flags { i: bool, m: bool, s: bool, x: bool }

#[derive(Clone, Copy, PartialEq)]
enum A { Bol, BolM, Eol, EolM, Begin, End, WordB, NotWordB }

#[derive(Clone)]
enum Node {
    Empty,
    Char(u8, bool),
    Any(bool),
    Class(u128),
    Assert(A),
    Group(Box<Node>, usize),
    Cat(Vec<Node>),
    Alt(Vec<Node>),
    Rep { n: Box<Node>, min: usize, max: Option<usize>, greedy: bool, poss: bool },
    Backref(usize, bool),
    Look { ahead: bool, neg: bool, n: Box<Node> },
    Atomic(Box<Node>),
}

fn is_word(c: u8) -> bool { c.is_ascii_alphanumeric() || c == b'_' }
fn set_range(m: &mut u128, a: u8, b: u8) { for c in a..=b { *m |= 1u128 << c; } }
fn class_d() -> u128 { let mut m = 0; set_range(&mut m, b'0', b'9'); m }
fn class_w() -> u128 { let mut m = 0; set_range(&mut m, b'0', b'9'); set_range(&mut m, b'a', b'z'); set_range(&mut m, b'A', b'Z'); m |= 1 << b'_'; m }
fn class_s() -> u128 { let mut m = 0; for c in [b' ', b'\t', b'\n', b'\r', 0x0b, 0x0c] { m |= 1u128 << c; } m }
fn negate(m: u128) -> u128 { !m }
fn fold_class(m: u128) -> u128 {
    let mut r = m;
    for c in b'a'..=b'z' {
        if m >> c & 1 == 1 || m >> (c - 32) & 1 == 1 { r |= 1u128 << c; r |= 1u128 << (c - 32); }
    }
    r
}

struct Parser<'a> { p: &'a [u8], i: usize, ngroups: usize, names: Vec<(String, usize)>, closed: Vec<bool> }

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<u8> { self.p.get(self.i).copied() }
    // parse {m} {m,} {m,n} {,n} at self.i; returns (min,max,len)
    fn brace(&self) -> Option<(usize, Option<usize>, usize)> {
        let s = &self.p[self.i..];
        if s.first() != Some(&b'{') { return None; }
        let mut j = 1;
        let a0 = j;
        while j < s.len() && s[j].is_ascii_digit() { j += 1; }
        let a: Option<usize> = if j > a0 { std::str::from_utf8(&s[a0..j]).unwrap().parse().ok() } else { None };
        if j < s.len() && s[j] == b'}' {
            return a.map(|v| (v, Some(v), j + 1));
        }
        if j < s.len() && s[j] == b',' {
            j += 1;
            let b0 = j;
            while j < s.len() && s[j].is_ascii_digit() { j += 1; }
            let b: Option<usize> = if j > b0 { std::str::from_utf8(&s[b0..j]).unwrap().parse().ok() } else { None };
            if j < s.len() && s[j] == b'}' {
                if a.is_none() && b.is_none() { return None; }
                return Some((a.unwrap_or(0), b, j + 1));
            }
        }
        None
    }

    fn alt(&mut self, f: Flags, depth: usize) -> R<Node> {
        let mut alts = vec![self.cat(f, depth)?];
        while self.peek() == Some(b'|') {
            self.i += 1;
            alts.push(self.cat(f, depth)?);
        }
        Ok(if alts.len() == 1 { alts.pop().unwrap() } else { Node::Alt(alts) })
    }

    fn cat(&mut self, mut f: Flags, depth: usize) -> R<Node> {
        let mut items: Vec<Node> = Vec::new();
        loop {
            let c = match self.peek() { None => break, Some(c) => c };
            if c == b'|' { break; }
            if c == b')' {
                if depth == 0 { return Err("paren"); }
                break;
            }
            if f.x {
                if c == b' ' || c == b'\t' || c == b'\n' || c == b'\r' || c == 0x0b || c == 0x0c { self.i += 1; continue; }
                if c == b'#' { self.i = self.p.len(); continue; }
            }
            let _ = &mut f;
            let atom = self.atom(f, depth)?;
            let atom = self.quant(atom, f)?;
            items.push(atom);
        }
        Ok(match items.len() { 0 => Node::Empty, 1 => items.pop().unwrap(), _ => Node::Cat(items) })
    }

    fn quant(&mut self, atom: Node, f: Flags) -> R<Node> {
        let mut atom = atom;
        let mut seen = false;
        loop {
            if f.x {
                while let Some(c) = self.peek() { if c == b' ' || c == b'\t' || c == b'\n' || c == b'\r' || c == 0x0b || c == 0x0c { self.i += 1; } else { break; } }
            }
            let (min, max, len) = match self.peek() {
                Some(b'*') => (0, None, 1),
                Some(b'+') => (1, None, 1),
                Some(b'?') => (0, Some(1), 1),
                Some(b'{') => match self.brace() { Some(t) => t, None => break },
                _ => break,
            };
            if seen { return Err("repeat"); }
            if matches!(atom, Node::Assert(_)) { return Err("repeat"); }
            if let Some(mx) = max { if min > mx { return Err("repeat"); } }
            self.i += len;
            let (mut greedy, mut poss) = (true, false);
            if self.peek() == Some(b'?') { greedy = false; self.i += 1; }
            else if self.peek() == Some(b'+') { poss = true; self.i += 1; }
            atom = Node::Rep { n: Box::new(atom), min, max, greedy, poss };
            seen = true;
        }
        Ok(atom)
    }

    fn hex(&mut self) -> R<u8> {
        // after \x
        let h = |c: u8| (c as char).to_digit(16);
        if self.i + 2 > self.p.len() { return Err("escape"); }
        match (h(self.p[self.i]), h(self.p[self.i + 1])) {
            (Some(a), Some(b)) => { self.i += 2; let v = (a * 16 + b) as u8; if v > 127 { Err("escape") } else { Ok(v) } }
            _ => Err("escape"),
        }
    }

    // escape that denotes a single literal byte; returns None when not a simple literal
    fn simple_escape(&mut self, c: u8) -> R<Option<u8>> {
        Ok(match c {
            b'n' => Some(b'\n'), b't' => Some(b'\t'), b'r' => Some(b'\r'), b'f' => Some(0x0c), b'v' => Some(0x0b),
            b'x' => Some(self.hex()?),
            c if !c.is_ascii_alphanumeric() => Some(c),
            _ => None,
        })
    }

    fn atom(&mut self, f: Flags, depth: usize) -> R<Node> {
        let c = self.p[self.i];
        self.i += 1;
        match c {
            b'*' | b'+' | b'?' => Err("repeat"),
            b'{' => {
                self.i -= 1;
                if self.brace().is_some() { return Err("repeat"); }
                self.i += 1;
                Ok(Node::Char(b'{', f.i))
            }
            b'.' => Ok(Node::Any(f.s)),
            b'^' => Ok(Node::Assert(if f.m { A::BolM } else { A::Bol })),
            b'$' => Ok(Node::Assert(if f.m { A::EolM } else { A::Eol })),
            b'[' => self.class(f),
            b'(' => self.group(f, depth),
            b'\\' => {
                let e = match self.peek() { Some(e) => e, None => return Err("escape") };
                self.i += 1;
                match e {
                    b'd' => Ok(Node::Class(class_d())), b'D' => Ok(Node::Class(negate(class_d()))),
                    b'w' => Ok(Node::Class(class_w())), b'W' => Ok(Node::Class(negate(class_w()))),
                    b's' => Ok(Node::Class(class_s())), b'S' => Ok(Node::Class(negate(class_s()))),
                    b'b' => Ok(Node::Assert(A::WordB)), b'B' => Ok(Node::Assert(A::NotWordB)),
                    b'A' => Ok(Node::Assert(A::Begin)), b'Z' => Ok(Node::Assert(A::End)),
                    b'1'..=b'9' => {
                        let g = (e - b'0') as usize;
                        if g > self.ngroups || !self.closed[g] { return Err("backref"); }
                        Ok(Node::Backref(g, f.i))
                    }
                    _ => match self.simple_escape(e)? { Some(b) => Ok(Node::Char(b, f.i)), None => Err("escape") },
                }
            }
            c => Ok(Node::Char(c, f.i)),
        }
    }

    fn class(&mut self, f: Flags) -> R<Node> {
        let mut neg = false;
        if self.peek() == Some(b'^') { neg = true; self.i += 1; }
        let mut m: u128 = 0;
        let mut first = true;
        loop {
            let c = match self.peek() { None => return Err("class"), Some(c) => c };
            self.i += 1;
            if c == b']' && !first { break; }
            first = false;
            // returns either a set or a single byte
            let lo: Result<u8, u128> = if c == b'\\' {
                let e = match self.peek() { None => return Err("class"), Some(e) => e };
                self.i += 1;
                match e {
                    b'd' => Err(class_d()), b'D' => Err(negate(class_d())), b'w' => Err(class_w()), b'W' => Err(negate(class_w())),
                    b's' => Err(class_s()), b'S' => Err(negate(class_s())),
                    b'b' => Ok(8),
                    _ => match self.simple_escape(e)? { Some(b) => Ok(b), None => return Err("escape") },
                }
            } else { Ok(c) };
            let lo = match lo { Err(set) => {
                // a set cannot start a range
                if self.peek() == Some(b'-') && self.p.get(self.i + 1).map_or(false, |&n| n != b']') { return Err("class"); }
                m |= set; continue; }
                Ok(b) => b };
            if self.peek() == Some(b'-') && self.p.get(self.i + 1).map_or(false, |&n| n != b']') {
                self.i += 1;
                let h = self.p[self.i]; self.i += 1;
                let hi = if h == b'\\' {
                    let e = match self.peek() { None => return Err("class"), Some(e) => e };
                    self.i += 1;
                    match e {
                        b'd' | b'D' | b'w' | b'W' | b's' | b'S' => return Err("class"),
                        b'b' => 8,
                        _ => match self.simple_escape(e)? { Some(b) => b, None => return Err("escape") },
                    }
                } else { h };
                if lo > hi { return Err("class"); }
                set_range(&mut m, lo, hi);
            } else { m |= 1u128 << lo; }
        }
        if f.i { m = fold_class(m); }
        if neg { m = negate(m); }
        Ok(Node::Class(m))
    }

    fn name(&mut self, term: u8) -> R<String> {
        let st = self.i;
        while let Some(c) = self.peek() { if c == term { break; } self.i += 1; }
        if self.peek() != Some(term) { return Err("name"); }
        let n = &self.p[st..self.i];
        self.i += 1;
        if n.is_empty() || !(n[0].is_ascii_alphabetic() || n[0] == b'_') || !n.iter().all(|&c| is_word(c)) { return Err("name"); }
        Ok(String::from_utf8(n.to_vec()).unwrap())
    }

    fn group(&mut self, f: Flags, depth: usize) -> R<Node> {
        if self.peek() != Some(b'?') {
            self.ngroups += 1;
            let g = self.ngroups;
            self.closed.push(false);
            let n = self.alt(f, depth + 1)?;
            if self.peek() != Some(b')') { return Err("paren"); }
            self.i += 1;
            self.closed[g] = true;
            return Ok(Node::Group(Box::new(n), g));
        }
        self.i += 1;
        let c = match self.peek() { None => return Err("group"), Some(c) => c };
        self.i += 1;
        let close = |s: &mut Parser| -> R<()> { if s.peek() != Some(b')') { return Err("paren"); } s.i += 1; Ok(()) };
        match c {
            b':' => { let n = self.alt(f, depth + 1)?; close(self)?; Ok(n_wrap(n)) }
            b'>' => { let n = self.alt(f, depth + 1)?; close(self)?; Ok(Node::Atomic(Box::new(n))) }
            b'=' | b'!' => { let n = self.alt(f, depth + 1)?; close(self)?; Ok(Node::Look { ahead: true, neg: c == b'!', n: Box::new(n) }) }
            b'<' => {
                let d = match self.peek() { Some(d @ (b'=' | b'!')) => d, _ => return Err("group") };
                self.i += 1;
                let n = self.alt(f, depth + 1)?; close(self)?;
                if width(&n).is_none() { return Err("lookbehind"); }
                Ok(Node::Look { ahead: false, neg: d == b'!', n: Box::new(n) })
            }
            b'P' => {
                match self.peek() {
                    Some(b'<') => {
                        self.i += 1;
                        let nm = self.name(b'>')?;
                        if self.names.iter().any(|(n, _)| *n == nm) { return Err("name"); }
                        self.ngroups += 1;
                        let g = self.ngroups;
                        self.closed.push(false);
                        self.names.push((nm, g));
                        let n = self.alt(f, depth + 1)?; close(self)?;
                        self.closed[g] = true;
                        Ok(Node::Group(Box::new(n), g))
                    }
                    Some(b'=') => {
                        self.i += 1;
                        let nm = self.name(b')')?;
                        match self.names.iter().find(|(n, _)| *n == nm) {
                            None => Err("name"),
                            Some(&(_, g)) => { if !self.closed[g] { return Err("backref"); } Ok(Node::Backref(g, f.i)) }
                        }
                    }
                    _ => Err("group"),
                }
            }
            _ if matches!(c, b'i' | b'm' | b's' | b'x' | b'-') => {
                self.i -= 1;
                let mut nf = f;
                let mut on = true;
                let mut any = false;
                loop {
                    let c = match self.peek() { None => return Err("flags"), Some(c) => c };
                    self.i += 1;
                    match c {
                        b'i' => nf.i = on, b'm' => nf.m = on, b's' => nf.s = on, b'x' => nf.x = on,
                        b'-' => { if !on { return Err("flags"); } on = false; any = true; }
                        b':' => {
                            let n = self.alt(nf, depth + 1)?; close(self)?;
                            return Ok(n_wrap(n));
                        }
                        b')' => {
                            let _ = any;
                            return Err("flags"); // global flags handled before parsing; here => not at start
                        }
                        _ => return Err("flags"),
                    }
                }
            }
            _ => Err("group"),
        }
    }
}

fn n_wrap(n: Node) -> Node { Node::Cat(vec![n]) }

fn width(n: &Node) -> Option<usize> {
    match n {
        Node::Empty | Node::Assert(_) | Node::Look { .. } => Some(0),
        Node::Char(..) | Node::Any(_) | Node::Class(_) => Some(1),
        Node::Group(b, _) | Node::Atomic(b) => width(b),
        Node::Cat(v) => { let mut t = 0; for x in v { t += width(x)?; } Some(t) }
        Node::Alt(v) => { let w = width(&v[0])?; for x in &v[1..] { if width(x)? != w { return None; } } Some(w) }
        Node::Rep { n, min, max, .. } => { if Some(*min) != *max { None } else { Some(width(n)? * min) } }
        Node::Backref(..) => None,
    }
}

fn nullable(n: &Node) -> bool {
    match n {
        Node::Empty | Node::Assert(_) | Node::Look { .. } | Node::Backref(..) => true,
        Node::Char(..) | Node::Any(_) | Node::Class(_) => false,
        Node::Group(b, _) | Node::Atomic(b) => nullable(b),
        Node::Cat(v) => v.iter().all(nullable),
        Node::Alt(v) => v.iter().any(nullable),
        Node::Rep { n, min, .. } => *min == 0 || nullable(n),
    }
}

#[derive(Clone)]
enum I {
    Char(u8), CharI(u8), Any, AnyNl, Class(u128), Split(usize, usize), Jmp(usize), Save(usize), Assert(A),
    Backref(usize, bool), Look { ahead: bool, neg: bool, sub: usize, width: usize }, Atomic(usize),
    Mark(usize), Check(usize, usize), Match,
}

struct Prog { progs: Vec<Vec<I>>, ngroups: usize, nslots: usize, memo_ok: bool }

struct Comp { progs: Vec<Vec<I>>, nmarks: usize, size: usize, special: bool, ngroups: usize }

impl Comp {
    fn emit(&mut self, n: &Node, p: usize) -> R<()> {
        self.size += 1;
        if self.size > 400_000 { return Err("repeat"); }
        match n {
            Node::Empty => {}
            Node::Char(c, icase) => {
                if *icase && c.is_ascii_alphabetic() { self.progs[p].push(I::CharI(c.to_ascii_lowercase())) } else { self.progs[p].push(I::Char(*c)) }
            }
            Node::Any(dotall) => self.progs[p].push(if *dotall { I::AnyNl } else { I::Any }),
            Node::Class(m) => self.progs[p].push(I::Class(*m)),
            Node::Assert(a) => self.progs[p].push(I::Assert(*a)),
            Node::Group(b, g) => {
                self.progs[p].push(I::Save(2 * g));
                self.emit(b, p)?;
                self.progs[p].push(I::Save(2 * g + 1));
            }
            Node::Cat(v) => for x in v { self.emit(x, p)?; },
            Node::Alt(v) => {
                let mut jmps = vec![];
                for (k, x) in v.iter().enumerate() {
                    if k + 1 < v.len() {
                        let s = self.progs[p].len();
                        self.progs[p].push(I::Split(s + 1, 0));
                        self.emit(x, p)?;
                        jmps.push(self.progs[p].len());
                        self.progs[p].push(I::Jmp(0));
                        let nxt = self.progs[p].len();
                        self.progs[p][s] = I::Split(s + 1, nxt);
                    } else { self.emit(x, p)?; }
                }
                let end = self.progs[p].len();
                for j in jmps { self.progs[p][j] = I::Jmp(end); }
            }
            Node::Backref(g, icase) => { self.special = true; self.progs[p].push(I::Backref(*g, *icase)); }
            Node::Look { ahead, neg, n } => {
                self.special = true;
                let sub = self.new_prog();
                self.emit(n, sub)?;
                self.progs[sub].push(I::Match);
                let w = if *ahead { 0 } else { width(n).unwrap() };
                self.progs[p].push(I::Look { ahead: *ahead, neg: *neg, sub, width: w });
            }
            Node::Atomic(b) => {
                self.special = true;
                let sub = self.new_prog();
                self.emit(b, sub)?;
                self.progs[sub].push(I::Match);
                self.progs[p].push(I::Atomic(sub));
            }
            Node::Rep { n, min, max, greedy, poss } => {
                if *poss {
                    self.special = true;
                    let sub = self.new_prog();
                    let inner = Node::Rep { n: n.clone(), min: *min, max: *max, greedy: true, poss: false };
                    self.emit(&inner, sub)?;
                    self.progs[sub].push(I::Match);
                    self.progs[p].push(I::Atomic(sub));
                    return Ok(());
                }
                for _ in 0..*min { self.emit(n, p)?; }
                match max {
                    None => {
                        let l = self.progs[p].len();
                        self.progs[p].push(I::Split(0, 0));
                        let null = nullable(n);
                        let mk = if null { self.special = true; self.nmarks += 1; Some(self.nmarks - 1) } else { None };
                        if let Some(k) = mk { self.progs[p].push(I::Mark(k)); }
                        self.emit(n, p)?;
                        let chk = self.progs[p].len();
                        if let Some(k) = mk { self.progs[p].push(I::Check(k, 0)); }
                        self.progs[p].push(I::Jmp(l));
                        let e = self.progs[p].len();
                        self.progs[p][l] = if *greedy { I::Split(l + 1, e) } else { I::Split(e, l + 1) };
                        if let Some(k) = mk { self.progs[p][chk] = I::Check(k, e); }
                    }
                    Some(mx) => {
                        let mut splits = vec![];
                        for _ in *min..*mx {
                            splits.push(self.progs[p].len());
                            self.progs[p].push(I::Split(0, 0));
                            self.emit(n, p)?;
                        }
                        let e = self.progs[p].len();
                        for s in splits { self.progs[p][s] = if *greedy { I::Split(s + 1, e) } else { I::Split(e, s + 1) }; }
                    }
                }
            }
        }
        Ok(())
    }
    fn new_prog(&mut self) -> usize { self.progs.push(vec![]); self.progs.len() - 1 }
}

fn compile(pat: &[u8], flags: Flags) -> R<(Prog, Vec<(String, usize)>)> {
    // global flags at the very start: (?imsx)
    let mut f = flags;
    let mut start = 0;
    if pat.len() >= 3 && pat[0] == b'(' && pat[1] == b'?' {
        let mut j = 2;
        while j < pat.len() && matches!(pat[j], b'i' | b'm' | b's' | b'x') { j += 1; }
        if j > 2 && j < pat.len() && pat[j] == b')' {
            for &c in &pat[2..j] { match c { b'i' => f.i = true, b'm' => f.m = true, b's' => f.s = true, _ => f.x = true } }
            start = j + 1;
        }
    }
    let mut ps = Parser { p: pat, i: start, ngroups: 0, names: vec![], closed: vec![false] };
    let node = ps.alt(f, 0)?;
    if ps.i < pat.len() { return Err("paren"); }
    let mut c = Comp { progs: vec![vec![]], nmarks: 0, size: 0, special: false, ngroups: ps.ngroups };
    c.emit(&node, 0)?;
    c.progs[0].push(I::Match);
    let nslots = 2 * (ps.ngroups + 1) + c.nmarks;
    let special = c.special;
    let ng = c.ngroups;
    Ok((Prog { progs: c.progs, ngroups: ng, nslots, memo_ok: !special }, ps.names))
}

enum Bt { Alt(usize, usize), Cap(usize, usize) }

struct Machine<'a> { pr: &'a Prog, t: &'a [u8], visited: Vec<bool>, memo: bool, steps: u64 }

impl<'a> Machine<'a> {
    fn assert(&self, a: A, pos: usize) -> bool {
        let t = self.t;
        match a {
            A::Bol | A::Begin => pos == 0,
            A::BolM => pos == 0 || t[pos - 1] == b'\n',
            A::Eol => pos == t.len() || (pos + 1 == t.len() && t[pos] == b'\n'),
            A::EolM => pos == t.len() || t[pos] == b'\n',
            A::End => pos == t.len(),
            A::WordB | A::NotWordB => {
                let a1 = pos > 0 && is_word(t[pos - 1]);
                let b1 = pos < t.len() && is_word(t[pos]);
                (a1 != b1) == (a == A::WordB)
            }
        }
    }

    // returns end position on success; caps updated
    fn run(&mut self, prog: usize, mut pos: usize, caps: &mut Vec<usize>, reject: Option<usize>, start: usize) -> Option<usize> {
        let code = &self.pr.progs[prog];
        let mut stack: Vec<Bt> = Vec::new();
        let mut pc = 0usize;
        let n1 = self.t.len() + 1;
        loop {
            self.steps += 1;
            let mut fail = false;
            match &code[pc] {
                I::Char(c) => { if pos < self.t.len() && self.t[pos] == *c { pos += 1; pc += 1; } else { fail = true; } }
                I::CharI(c) => { if pos < self.t.len() && self.t[pos].to_ascii_lowercase() == *c { pos += 1; pc += 1; } else { fail = true; } }
                I::Any => { if pos < self.t.len() && self.t[pos] != b'\n' { pos += 1; pc += 1; } else { fail = true; } }
                I::AnyNl => { if pos < self.t.len() { pos += 1; pc += 1; } else { fail = true; } }
                I::Class(m) => { if pos < self.t.len() && self.t[pos] < 128 && (m >> self.t[pos]) & 1 == 1 { pos += 1; pc += 1; } else { fail = true; } }
                I::Split(a, b) => {
                    if self.memo && prog == 0 {
                        let k = pc * n1 + pos;
                        if self.visited[k] { fail = true; } else { self.visited[k] = true; stack.push(Bt::Alt(*b, pos)); pc = *a; }
                    } else { stack.push(Bt::Alt(*b, pos)); pc = *a; }
                }
                I::Jmp(a) => pc = *a,
                I::Save(i) => { stack.push(Bt::Cap(*i, caps[*i])); caps[*i] = pos; pc += 1; }
                I::Mark(k) => { let i = 2 * (self.pr.ngroups + 1) + k; stack.push(Bt::Cap(i, caps[i])); caps[i] = pos; pc += 1; }
                I::Check(k, ex) => { let i = 2 * (self.pr.ngroups + 1) + k; if caps[i] == pos { pc = *ex; } else { pc += 1; } }
                I::Assert(a) => { if self.assert(*a, pos) { pc += 1; } else { fail = true; } }
                I::Backref(g, ic) => {
                    let (s, e) = (caps[2 * g], caps[2 * g + 1]);
                    if s == NONE || e == NONE { fail = true; } else {
                        let l = e - s;
                        if pos + l > self.t.len() { fail = true; } else {
                            let ok = (0..l).all(|k| if *ic { self.t[s + k].to_ascii_lowercase() == self.t[pos + k].to_ascii_lowercase() } else { self.t[s + k] == self.t[pos + k] });
                            if ok { pos += l; pc += 1; } else { fail = true; }
                        }
                    }
                }
                I::Look { ahead, neg, sub, width } => {
                    let before = caps.clone();
                    let ok = if *ahead { self.run(*sub, pos, caps, None, pos).is_some() }
                             else if pos >= *width { self.run(*sub, pos - *width, caps, None, pos - *width) == Some(pos) } else { false };
                    if *neg {
                        *caps = before;
                        if ok { fail = true; } else { pc += 1; }
                    } else if ok {
                        for i in 0..caps.len() { if caps[i] != before[i] { stack.push(Bt::Cap(i, before[i])); } }
                        pc += 1;
                    } else { *caps = before; fail = true; }
                }
                I::Atomic(sub) => {
                    let before = caps.clone();
                    match self.run(*sub, pos, caps, None, pos) {
                        Some(e) => {
                            for i in 0..caps.len() { if caps[i] != before[i] { stack.push(Bt::Cap(i, before[i])); } }
                            pos = e; pc += 1;
                        }
                        None => { *caps = before; fail = true; }
                    }
                }
                I::Match => {
                    if prog == 0 && reject == Some(start) && pos == start { fail = true; } else { return Some(pos); }
                }
            }
            if fail {
                loop {
                    match stack.pop() {
                        None => return None,
                        Some(Bt::Cap(i, v)) => caps[i] = v,
                        Some(Bt::Alt(p2, pos2)) => { pc = p2; pos = pos2; break; }
                    }
                }
            }
        }
    }
}

struct Re { pr: Prog, names: Vec<(String, usize)> }

impl Re {
    fn search(&self, t: &[u8], from: usize, reject: Option<usize>) -> Option<Vec<usize>> {
        let memo = self.pr.memo_ok;
        let mut m = Machine { pr: &self.pr, t, visited: if memo { vec![false; self.pr.progs[0].len() * (t.len() + 1)] } else { vec![] }, memo, steps: 0 };
        for start in from..=t.len() {
            let mut caps = vec![NONE; self.pr.nslots];
            caps[0] = start;
            if let Some(e) = m.run(0, start, &mut caps, reject, start) {
                caps[1] = e;
                return Some(caps);
            }
        }
        None
    }
    fn all(&self, t: &[u8]) -> Vec<Vec<usize>> {
        let mut out = vec![];
        let mut pos = 0;
        let mut prev_empty = false;
        while pos <= t.len() {
            match self.search(t, pos, if prev_empty { Some(pos) } else { None }) {
                None => break,
                Some(c) => { prev_empty = c[1] == c[0]; pos = c[1]; out.push(c); }
            }
        }
        out
    }
}

fn unescape(s: &str) -> Vec<u8> {
    let b = s.as_bytes();
    let mut out = vec![];
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'\\' && i + 1 < b.len() {
            match b[i + 1] {
                b'n' => { out.push(b'\n'); i += 2; }
                b't' => { out.push(b'\t'); i += 2; }
                b'r' => { out.push(b'\r'); i += 2; }
                b'\\' => { out.push(b'\\'); i += 2; }
                b'x' if i + 3 < b.len() + 0 && i + 3 <= b.len() - 1 + 1 => {
                    let h = std::str::from_utf8(&b[i + 2..(i + 4).min(b.len())]).unwrap_or("00");
                    out.push(u8::from_str_radix(h, 16).unwrap_or(0)); i += 4;
                }
                c => { out.push(b'\\'); out.push(c); i += 2; }
            }
        } else { out.push(b[i]); i += 1; }
    }
    out
}

fn encode(t: &[u8]) -> String {
    let mut s = String::new();
    for &c in t {
        match c {
            b'\\' => s.push_str("\\\\"), b'\n' => s.push_str("\\n"), b'\t' => s.push_str("\\t"), b'\r' => s.push_str("\\r"),
            0x20..=0x7e => s.push(c as char),
            _ => s.push_str(&format!("\\x{:02x}", c)),
        }
    }
    s
}

fn parse_flags(s: &str) -> Flags {
    Flags { i: s.contains('i'), m: s.contains('m'), s: s.contains('s'), x: s.contains('x') }
}

enum Tok { Lit(Vec<u8>), Grp(usize) }

fn template(t: &[u8], re: &Re) -> R<Vec<Tok>> {
    let mut out = vec![];
    let mut lit = vec![];
    let mut i = 0;
    while i < t.len() {
        if t[i] != b'\\' { lit.push(t[i]); i += 1; continue; }
        if i + 1 >= t.len() { return Err("template"); }
        let c = t[i + 1];
        match c {
            b'n' => { lit.push(b'\n'); i += 2; } b't' => { lit.push(b'\t'); i += 2; } b'r' => { lit.push(b'\r'); i += 2; }
            b'\\' => { lit.push(b'\\'); i += 2; }
            b'1'..=b'9' => {
                let g = (c - b'0') as usize;
                if g > re.pr.ngroups { return Err("template"); }
                if !lit.is_empty() { out.push(Tok::Lit(std::mem::take(&mut lit))); }
                out.push(Tok::Grp(g)); i += 2;
            }
            b'g' => {
                if t.get(i + 2) != Some(&b'<') { return Err("template"); }
                let end = match t[i + 3..].iter().position(|&x| x == b'>') { Some(e) => i + 3 + e, None => return Err("template") };
                let nm = std::str::from_utf8(&t[i + 3..end]).map_err(|_| "template")?;
                let g = if let Ok(n) = nm.parse::<usize>() { n } else {
                    match re.names.iter().find(|(n, _)| n == nm) { Some(&(_, g)) => g, None => return Err("template") }
                };
                if g > re.pr.ngroups { return Err("template"); }
                if !lit.is_empty() { out.push(Tok::Lit(std::mem::take(&mut lit))); }
                out.push(Tok::Grp(g)); i = end + 1;
            }
            _ => return Err("template"),
        }
    }
    if !lit.is_empty() { out.push(Tok::Lit(lit)); }
    Ok(out)
}

fn span(c: &[usize], g: usize) -> String {
    if c[2 * g] == NONE || c[2 * g + 1] == NONE { "-".into() } else { format!("{},{}", c[2 * g], c[2 * g + 1]) }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() != 3 { eprintln!("usage: rx match|findall|sub FILE"); std::process::exit(2); }
    let data = fs::read(&a[2]).unwrap_or_else(|_| { eprintln!("rx: cannot read {}", a[2]); std::process::exit(2) });
    let text = String::from_utf8_lossy(&data).to_string();
    let lines: Vec<&str> = text.split('\n').collect();
    let per = if a[1] == "sub" { 4 } else { 3 };
    let ncases = (lines.len().saturating_sub(1)) / per;
    let out = std::io::stdout();
    let mut out = out.lock();
    for k in 0..ncases {
        let l = &lines[k * per..(k + 1) * per];
        let pat = l[0].as_bytes();
        let flags = parse_flags(l[1]);
        let (tmpl, subj) = if per == 4 { (Some(l[2]), l[3]) } else { (None, l[2]) };
        let subj = unescape(subj);
        let line = match compile(pat, flags) {
            Err(e) => format!("E {}", e),
            Ok((pr, names)) => {
                let re = Re { pr, names };
                match a[1].as_str() {
                    "match" => match re.search(&subj, 0, None) {
                        None => "N".to_string(),
                        Some(c) => {
                            let mut s = String::from("M");
                            for g in 0..=re.pr.ngroups { s.push(' '); s.push_str(&span(&c, g)); }
                            s
                        }
                    },
                    "findall" => {
                        let all = re.all(&subj);
                        let mut s = format!("F {}", all.len());
                        for c in &all { s.push_str(&format!(" {},{}", c[0], c[1])); }
                        s
                    }
                    _ => match template(tmpl.unwrap().as_bytes(), &re) {
                        Err(e) => format!("E {}", e),
                        Ok(toks) => {
                            let all = re.all(&subj);
                            let mut res: Vec<u8> = vec![];
                            let mut last = 0;
                            for c in &all {
                                res.extend_from_slice(&subj[last..c[0]]);
                                for t in &toks {
                                    match t {
                                        Tok::Lit(b) => res.extend_from_slice(b),
                                        Tok::Grp(g) => if c[2 * g] != NONE && c[2 * g + 1] != NONE { res.extend_from_slice(&subj[c[2 * g]..c[2 * g + 1]]); },
                                    }
                                }
                                last = c[1];
                            }
                            res.extend_from_slice(&subj[last..]);
                            format!("S {}", encode(&res))
                        }
                    },
                }
            }
        };
        writeln!(out, "{}", line).unwrap();
    }
}
