'use strict';
// Readable source of the re-bundle target ("tool"). NOT visible to the agent: only the obfuscated bundle is.
const fs = require('fs');

function fail(msg, code) { process.stderr.write(msg + '\n'); process.exit(code === undefined ? 1 : code); }
function out(buf) {
  const b = Buffer.isBuffer(buf) ? buf : Buffer.from(String(buf), 'utf8');
  let o = 0;
  while (o < b.length) o += fs.writeSync(1, b, o, b.length - o);
}

function readInput(arg, who) {
  try { return fs.readFileSync(arg === undefined || arg === '-' ? 0 : arg); }
  catch (e) { if (arg === undefined || arg === '-') return Buffer.alloc(0); fail(who + ': cannot read ' + arg, 2); }
}

// ---------------------------------------------------------------- checksums
function crcOf(bytes) {
  let h = 0x811c9dc5;
  for (let i = 0; i < bytes.length; i++) {
    h = (h ^ bytes[i]) >>> 0;
    h = Math.imul(h, 0x01000193) >>> 0;
    h = ((h << 5) | (h >>> 27)) >>> 0;
  }
  return (h ^ bytes.length) >>> 0;
}
function adlerOf(bytes) {
  let a = 1, b = 0;
  for (let i = 0; i < bytes.length; i++) { a = (a + bytes[i]) % 65521; b = (b + a) % 65521; }
  return ((b << 16) | a) >>> 0;
}
function hex8(n) { return (n >>> 0).toString(16).padStart(8, '0'); }

function cmdCrc(args) {
  let data;
  if (args[0] === '-s') { if (args.length < 2) fail('usage: tool crc [-s STRING | FILE]', 2); data = Buffer.from(args[1], 'utf8'); }
  else data = readInput(args[0], 'crc');
  out('crc=' + hex8(crcOf(data)) + ' adler=' + hex8(adlerOf(data)) + ' len=' + data.length + '\n');
}

// ---------------------------------------------------------------- b32
const ALPHA = '0123456789BCDFGHJKLMNPQRSTVWXYZ_';
function cmdB32(args) {
  const mode = args[0];
  if (mode !== 'enc' && mode !== 'dec') fail('usage: tool b32 enc|dec [FILE]', 2);
  const data = readInput(args[1], 'b32');
  if (mode === 'enc') {
    let acc = 0, bits = 0, s = '', sum = 0;
    for (let i = 0; i < data.length; i++) {
      acc = ((acc << 8) | data[i]) & 0xffff; bits += 8;
      while (bits >= 5) { const v = (acc >> (bits - 5)) & 31; s += ALPHA[v]; sum += v; bits -= 5; }
    }
    if (bits > 0) { const v = (acc << (5 - bits)) & 31; s += ALPHA[v]; sum += v; }
    const groups = [];
    for (let i = 0; i < s.length; i += 5) groups.push(s.slice(i, i + 5));
    out(groups.join('-') + '~' + ALPHA[(sum + data.length) % 32] + '\n');
    return;
  }
  const text = data.toString('utf8').trim();
  const t = text.indexOf('~');
  if (t < 0 || text.indexOf('~', t + 1) >= 0) fail('b32: missing checksum');
  const body = text.slice(0, t), chk = text.slice(t + 1);
  const vals = [];
  for (let i = 0; i < body.length; i++) {
    const c = body[i];
    if (c === '-') continue;
    const v = ALPHA.indexOf(c.toUpperCase());
    if (v < 0) fail("b32: invalid character '" + c + "' at " + i);
    vals.push(v);
  }
  const nbytes = Math.floor(vals.length * 5 / 8), extra = vals.length * 5 - nbytes * 8;
  if (extra >= 5) fail('b32: bad length');
  const res = Buffer.alloc(nbytes);
  let acc = 0, bits = 0, o = 0, sum = 0;
  for (const v of vals) {
    sum += v; acc = ((acc << 5) | v) & 0xffff; bits += 5;
    if (bits >= 8) { if (o < nbytes) res[o++] = (acc >> (bits - 8)) & 255; bits -= 8; }
  }
  if (bits > 0 && (acc & ((1 << bits) - 1)) !== 0) fail('b32: nonzero padding');
  if (chk.length !== 1 || ALPHA[(sum + nbytes) % 32] !== chk.toUpperCase()) fail('b32: bad checksum');
  out(res);
}

// ---------------------------------------------------------------- ser / de
function serVal(v, parts) {
  if (v === null) parts.push(Buffer.from('n'));
  else if (v === true) parts.push(Buffer.from('t'));
  else if (v === false) parts.push(Buffer.from('f'));
  else if (typeof v === 'number') {
    if (!Number.isSafeInteger(v)) fail('ser: unsupported number');
    parts.push(Buffer.from('i' + String(v) + ';'));
  } else if (typeof v === 'string') serStr(v, parts);
  else if (Array.isArray(v)) { parts.push(Buffer.from('[')); for (const x of v) serVal(x, parts); parts.push(Buffer.from(']')); }
  else {
    parts.push(Buffer.from('{'));
    for (const k of Object.keys(v).sort()) { serStr(k, parts); serVal(v[k], parts); }
    parts.push(Buffer.from('}'));
  }
}
function serStr(s, parts) { const b = Buffer.from(s, 'utf8'); parts.push(Buffer.from('s' + b.length + ':')); parts.push(b); }

function cmdSer(args) {
  const text = readInput(args[0], 'ser').toString('utf8');
  let v;
  try { v = JSON.parse(text); } catch (e) { fail('ser: invalid JSON'); }
  const parts = []; serVal(v, parts);
  const body = Buffer.concat(parts);
  out(Buffer.concat([body, Buffer.from('#' + hex8(crcOf(body)) + '\n')]));
}

function cmdDe(args) {
  const raw = readInput(args[0], 'de');
  let end = raw.length;
  while (end > 0 && (raw[end - 1] === 10 || raw[end - 1] === 13 || raw[end - 1] === 32)) end--;
  const data = raw.subarray(0, end);
  const hp = data.lastIndexOf(35);
  if (hp < 0) fail('de: missing checksum');
  const field = data.subarray(hp + 1).toString('latin1');
  if (!/^[0-9a-f]{8}$/.test(field)) fail('de: bad checksum field');
  const body = data.subarray(0, hp);
  if (hex8(crcOf(body)) !== field) fail('de: checksum mismatch');
  let p = 0;
  function bad(msg) { fail('de: ' + msg + ' at ' + p); }
  function unexpected() {
    if (p >= body.length) bad('unexpected end');
    fail("de: unexpected '" + String.fromCharCode(body[p]) + "' at " + p);
  }
  function parse(depth) {
    if (depth > 20) fail('de: nesting too deep');
    if (p >= body.length) unexpected();
    const c = String.fromCharCode(body[p]);
    if (c === 'n') { p++; return null; }
    if (c === 't') { p++; return true; }
    if (c === 'f') { p++; return false; }
    if (c === 'i') {
      p++; const st = p;
      if (body[p] === 45) p++;
      const ds = p;
      while (p < body.length && body[p] >= 48 && body[p] <= 57) p++;
      if (p === ds || body[p] !== 59) { p = st; bad('bad integer'); }
      if (p - ds > 15) { p = st; bad('integer too large'); }
      const n = Number(body.subarray(st, p).toString('latin1')); p++;
      return n;
    }
    if (c === 's') return parseStr();
    if (c === '[') {
      p++; const arr = [];
      while (true) { if (p >= body.length) unexpected(); if (body[p] === 93) { p++; return arr; } arr.push(parse(depth + 1)); }
    }
    if (c === '{') {
      p++; const obj = {};
      while (true) {
        if (p >= body.length) unexpected();
        if (body[p] === 125) { p++; return obj; }
        if (body[p] !== 115) unexpected();
        const k = parseStr();
        obj[k] = parse(depth + 1);
      }
    }
    unexpected();
  }
  function parseStr() {
    p++; const st = p;
    while (p < body.length && body[p] >= 48 && body[p] <= 57) p++;
    if (p === st || body[p] !== 58) { p = st; bad('bad string length'); }
    const len = Number(body.subarray(st, p).toString('latin1')); p++;
    if (p + len > body.length) bad('string overflows');
    const s = body.subarray(p, p + len).toString('utf8'); p += len;
    return s;
  }
  const v = parse(0);
  if (p !== body.length) bad('trailing data');
  out(JSON.stringify(v) + '\n');
}

// ---------------------------------------------------------------- expr
function exprEval(src, vars) {
  let pos = 0, tok = null;
  const FUN = { min: [1, 99], max: [1, 99], abs: [1, 1], clamp: [3, 3], pow: [2, 2], sign: [1, 1] };
  function lex() {
    while (pos < src.length && ' \t'.indexOf(src[pos]) >= 0) pos++;
    const st = pos;
    if (pos >= src.length) return { t: 'end', s: '', p: st };
    const c = src[pos];
    if (/[0-9]/.test(c)) {
      const m = /^(0[xX][0-9a-fA-F]+|[0-9]+)/.exec(src.slice(pos));
      pos += m[0].length;
      if (m[0].length > 15) fail('expr: number too large at ' + st);
      return { t: 'num', v: Number(m[0]) | 0, p: st };
    }
    if (/[A-Za-z_]/.test(c)) { const m = /^[A-Za-z_][A-Za-z0-9_]*/.exec(src.slice(pos)); pos += m[0].length; return { t: 'id', s: m[0], p: st }; }
    const three = src.slice(pos, pos + 2);
    for (const op of ['<<', '>>', '<=', '>=', '==', '!=', '&&', '||']) if (three === op) { pos += 2; return { t: 'op', s: op, p: st }; }
    if ('+-*/%&|^~!<>?:(),'.indexOf(c) >= 0) { pos++; return { t: 'op', s: c, p: st }; }
    fail('expr: syntax error at ' + st);
  }
  function next() { tok = lex(); }
  function syn() { fail('expr: syntax error at ' + tok.p); }
  function isOp(s) { return tok.t === 'op' && tok.s === s; }
  // AST nodes are closures evaluated lazily
  function ternary() {
    const c = orx();
    if (isOp('?')) {
      next(); const a = ternary();
      if (!isOp(':')) syn();
      next(); const b = ternary();
      return () => (c() !== 0 ? a() : b());
    }
    return c;
  }
  function orx() { let l = andx(); while (isOp('||')) { next(); const a = l, r = andx(); l = () => ((a() !== 0 || r() !== 0) ? 1 : 0); } return l; }
  function andx() { let l = bor(); while (isOp('&&')) { next(); const a = l, r = bor(); l = () => ((a() !== 0 && r() !== 0) ? 1 : 0); } return l; }
  function bin(sub, ops, f) {
    return function () {
      let l = sub();
      while (tok.t === 'op' && ops.indexOf(tok.s) >= 0) { const op = tok.s; next(); const a = l, r = sub(); l = () => f(op, a(), r()); }
      return l;
    };
  }
  const mul = () => bin(unary, ['*', '/', '%'], (op, a, b) => {
    if (op === '*') return Math.imul(a, b);
    if (b === 0) fail('expr: division by zero');
    return op === '/' ? (Math.trunc(a / b) | 0) : ((a % b) | 0);
  })();
  const add = () => bin(mul, ['+', '-'], (op, a, b) => (op === '+' ? (a + b) | 0 : (a - b) | 0))();
  const shf = () => bin(add, ['<<', '>>'], (op, a, b) => (op === '<<' ? a << (b & 31) : a >> (b & 31)))();
  const rel = () => bin(shf, ['<', '<=', '>', '>='], (op, a, b) => (op === '<' ? a < b : op === '<=' ? a <= b : op === '>' ? a > b : a >= b) ? 1 : 0)();
  const eq = () => bin(rel, ['==', '!='], (op, a, b) => ((op === '==') === (a === b)) ? 1 : 0)();
  const band = () => bin(eq, ['&'], (op, a, b) => a & b)();
  const bxor = () => bin(band, ['^'], (op, a, b) => a ^ b)();
  const bor = () => bin(bxor, ['|'], (op, a, b) => a | b)();
  function unary() {
    if (isOp('-')) { next(); const a = unary(); return () => (-a()) | 0; }
    if (isOp('!')) { next(); const a = unary(); return () => (a() === 0 ? 1 : 0); }
    if (isOp('~')) { next(); const a = unary(); return () => ~a(); }
    return primary();
  }
  function primary() {
    if (tok.t === 'num') { const v = tok.v; next(); return () => v; }
    if (isOp('(')) { next(); const e = ternary(); if (!isOp(')')) syn(); next(); return e; }
    if (tok.t === 'id') {
      const name = tok.s; next();
      if (isOp('(')) {
        if (!(name in FUN) || !Object.prototype.hasOwnProperty.call(FUN, name)) fail("expr: unknown function '" + name + "'");
        next(); const args = [];
        if (!isOp(')')) { args.push(ternary()); while (isOp(',')) { next(); args.push(ternary()); } }
        if (!isOp(')')) syn();
        next();
        const [lo, hi] = FUN[name];
        if (args.length < lo || args.length > hi) fail("expr: wrong number of arguments for '" + name + "'");
        return () => {
          const v = args.map(f => f());
          switch (name) {
            case 'min': return Math.min(...v);
            case 'max': return Math.max(...v);
            case 'abs': return Math.abs(v[0]) | 0;
            case 'sign': return v[0] > 0 ? 1 : v[0] < 0 ? -1 : 0;
            case 'clamp': return Math.min(Math.max(v[0], v[1]), v[2]);
            default: {
              if (v[1] < 0) fail('expr: negative exponent');
              let r = 1, bs = v[0], e = v[1]; while (e > 0) { if (e & 1) r = Math.imul(r, bs); bs = Math.imul(bs, bs); e >>= 1; } return r;
            }
          }
        };
      }
      return () => {
        if (!Object.prototype.hasOwnProperty.call(vars, name)) fail("expr: unknown variable '" + name + "'");
        return vars[name];
      };
    }
    syn();
  }
  next();
  const e = ternary();
  if (tok.t !== 'end') syn();
  return e();
}
function cmdExpr(args) {
  if (args.length < 1) fail('usage: tool expr EXPR [name=value...]', 2);
  const vars = {};
  for (const a of args.slice(1)) {
    const m = /^([A-Za-z_][A-Za-z0-9_]*)=(-?[0-9]+)$/.exec(a);
    if (!m || m[2].length > 10) fail("expr: bad assignment '" + a + "'", 2);
    vars[m[1]] = Number(m[2]) | 0;
  }
  out(String(exprEval(args[0], vars)) + '\n');
}

// ---------------------------------------------------------------- tpl
function isTruthy(v) { return !(v === undefined || v === null || v === false || v === '' || v === 0); }
function show(v) {
  if (v === undefined || v === null) return '';
  if (Array.isArray(v)) return v.map(show).join(',');
  if (typeof v === 'object') return '[object Object]';
  return String(v);
}
function lookup(o, k) {
  if (typeof o === 'string') { if (k === 'length') return o.length; if (typeof k === 'number') return o[k]; return undefined; }
  if (Array.isArray(o)) { if (k === 'length') return o.length; if (typeof k === 'number') return o[k]; return undefined; }
  if (o !== null && typeof o === 'object') return Object.prototype.hasOwnProperty.call(o, k) ? o[k] : undefined;
  return undefined;
}
const FILTERS = {
  upper: v => show(v).toUpperCase(), lower: v => show(v).toLowerCase(), trim: v => show(v).trim(),
  len: v => (typeof v === 'string' || Array.isArray(v)) ? v.length : (v !== null && typeof v === 'object') ? Object.keys(v).length : 0,
  join: (v, a) => Array.isArray(v) ? v.map(show).join(a === undefined ? ',' : String(a)) : show(v),
  default: (v, a) => (v === undefined || v === null || v === '') ? a : v,
  rev: v => Array.isArray(v) ? v.slice().reverse() : show(v).split('').reverse().join(''),
};
function compileExpr(src, tagPos) {
  let i = 0;
  const synErr = () => fail('tpl: syntax error in tag at ' + tagPos);
  function ws() { while (i < src.length && /\s/.test(src[i])) i++; }
  function word() { ws(); const m = /^[A-Za-z_][A-Za-z0-9_]*/.exec(src.slice(i)); return m ? m[0] : null; }
  function orE() { let l = andE(); while (word() === 'or') { i += 2; const a = l, r = andE(); l = sc => (isTruthy(a(sc)) || isTruthy(r(sc))); } return l; }
  function andE() { let l = notE(); while (word() === 'and') { i += 3; const a = l, r = notE(); l = sc => (isTruthy(a(sc)) && isTruthy(r(sc))); } return l; }
  function notE() { if (word() === 'not') { i += 3; const a = notE(); return sc => !isTruthy(a(sc)); } return cmp(); }
  function cmp() {
    const a = operand(); ws();
    const m = /^(==|!=|<|>)/.exec(src.slice(i));
    if (!m) return a;
    i += m[0].length; const b = operand();
    return sc => {
      const x = a(sc), y = b(sc);
      if (m[0] === '==') return x === y;
      if (m[0] === '!=') return x !== y;
      if (typeof x !== typeof y || (typeof x !== 'number' && typeof x !== 'string')) return false;
      return m[0] === '<' ? x < y : x > y;
    };
  }
  function operand() {
    let v = primary();
    ws();
    while (src[i] === '|') {
      i++; const name = word();
      if (name === null) synErr();
      i += name.length;
      if (!Object.prototype.hasOwnProperty.call(FILTERS, name)) fail("tpl: unknown filter '" + name + "' at " + tagPos);
      let arg;
      ws();
      if (src[i] === '(') { i++; const lit = literal(); if (lit === undefined) synErr(); arg = lit.v; ws(); if (src[i] !== ')') synErr(); i++; }
      const f = FILTERS[name], prev = v;
      v = sc => f(prev(sc), arg);
      ws();
    }
    return v;
  }
  function literal() {
    ws();
    let m;
    if ((m = /^"([^"]*)"/.exec(src.slice(i))) || (m = /^'([^']*)'/.exec(src.slice(i)))) { i += m[0].length; return { v: m[1] }; }
    if ((m = /^-?[0-9]+/.exec(src.slice(i)))) { i += m[0].length; return { v: Number(m[0]) }; }
    return undefined;
  }
  function primary() {
    const lit = literal();
    if (lit !== undefined) return () => lit.v;
    const w = word();
    if (w === null) synErr();
    i += w.length;
    if (w === 'true') return () => true;
    if (w === 'false') return () => false;
    if (w === 'null') return () => null;
    const path = [{ k: w }];
    while (true) {
      if (src[i] === '.') {
        i++; const m = /^[A-Za-z_][A-Za-z0-9_]*/.exec(src.slice(i));
        if (!m) synErr();
        i += m[0].length; path.push({ k: m[0] });
      } else if (src[i] === '[') {
        i++; const ix = orE(); ws(); if (src[i] !== ']') synErr(); i++; path.push({ f: ix });
      } else break;
    }
    return sc => {
      let cur = sc(path[0].k);
      for (let j = 1; j < path.length; j++) cur = lookup(cur, path[j].f ? path[j].f(sc) : path[j].k);
      return cur;
    };
  }
  const e = orE(); ws();
  if (i < src.length) synErr();
  return e;
}
function compileTpl(text) {
  const root = { kind: 'seq', items: [] };
  const stack = [{ node: root, open: null }];
  let pos = 0, trimNext = false;
  function pushText(t) {
    if (trimNext) { t = t.replace(/^\s+/, ''); trimNext = false; }
    if (t) stack[stack.length - 1].node.items.push({ kind: 'text', s: t });
  }
  function cur() { return stack[stack.length - 1]; }
  while (pos < text.length) {
    const o = text.indexOf('{', pos);
    let found = -1;
    for (let q = o; q >= 0 && q < text.length; q = text.indexOf('{', q + 1)) {
      if ('{%#'.indexOf(text[q + 1]) >= 0 && q + 1 < text.length) { found = q; break; }
    }
    if (found < 0) { pushText(text.slice(pos)); break; }
    let lead = text.slice(pos, found);
    const kindc = text[found + 1];
    const closer = kindc === '{' ? '}}' : kindc === '%' ? '%}' : '#}';
    const e = text.indexOf(closer, found + 2);
    if (e < 0) fail('tpl: unterminated tag at ' + found);
    let inner = text.slice(found + 2, e);
    if (inner[0] === '-') { lead = lead.replace(/\s+$/, ''); inner = inner.slice(1); }
    let trimAfter = false;
    if (inner[inner.length - 1] === '-') { trimAfter = true; inner = inner.slice(0, -1); }
    pushText(lead);
    pos = e + 2;
    if (kindc === '#') { trimNext = trimAfter; continue; }
    if (kindc === '{') {
      cur().node.items.push({ kind: 'out', f: compileExpr(inner, found) });
    } else {
      const m = /^\s*([A-Za-z_]+)\s*([\s\S]*)$/.exec(inner);
      const kw = m ? m[1] : '';
      const rest = m ? m[2] : '';
      if (kw === 'if') {
        const n = { kind: 'if', cond: compileExpr(rest, found), a: { kind: 'seq', items: [] }, b: null };
        cur().node.items.push(n); stack.push({ node: n.a, open: 'endif', owner: n });
      } else if (kw === 'else') {
        const c = cur();
        if (c.open !== 'endif' || c.owner.b) fail("tpl: unexpected 'else' at " + found);
        c.owner.b = { kind: 'seq', items: [] }; c.node = c.owner.b;
      } else if (kw === 'for') {
        const fm = /^([A-Za-z_][A-Za-z0-9_]*)\s+in\s+([\s\S]+)$/.exec(rest);
        if (!fm) fail('tpl: syntax error in tag at ' + found);
        const n = { kind: 'for', v: fm[1], f: compileExpr(fm[2], found), body: { kind: 'seq', items: [] } };
        cur().node.items.push(n); stack.push({ node: n.body, open: 'endfor', owner: n });
      } else if (kw === 'endif' || kw === 'endfor') {
        if (rest.trim() !== '') fail('tpl: syntax error in tag at ' + found);
        if (cur().open !== kw) fail("tpl: unexpected '" + kw + "' at " + found);
        stack.pop();
      } else fail('tpl: syntax error in tag at ' + found);
    }
    trimNext = trimAfter;
  }
  if (stack.length > 1) fail("tpl: missing '" + cur().open + "'");
  return root;
}
function renderTpl(node, scope, buf) {
  for (const it of node.items) {
    if (it.kind === 'text') buf.push(it.s);
    else if (it.kind === 'out') buf.push(show(it.f(scope)));
    else if (it.kind === 'if') { if (isTruthy(it.cond(scope))) renderTpl(it.a, scope, buf); else if (it.b) renderTpl(it.b, scope, buf); }
    else {
      let seq = it.f(scope);
      if (typeof seq === 'string') seq = Array.from(seq);
      if (!Array.isArray(seq)) continue;
      seq.forEach((x, idx) => {
        const loop = { index: idx + 1, index0: idx, first: idx === 0, last: idx === seq.length - 1, length: seq.length };
        const inner = k => (k === it.v ? x : k === 'loop' ? loop : scope(k));
        renderTpl(it.body, inner, buf);
      });
    }
  }
}
function cmdTpl(args) {
  if (args.length < 2) fail('usage: tool tpl TEMPLATE DATA.json', 2);
  const text = readInput(args[0], 'tpl').toString('utf8');
  let data;
  try { data = JSON.parse(readInput(args[1], 'tpl').toString('utf8')); } catch (e) { fail('tpl: invalid JSON'); }
  const tree = compileTpl(text);
  const buf = [];
  renderTpl(tree, k => lookup(data, k), buf);
  out(buf.join(''));
}

// ---------------------------------------------------------------- pack / unpack (LZW, variable width)
function cmdPack(args) {
  const data = readInput(args[0], 'pack');
  const outb = [];
  let acc = 0, nb = 0;
  function emit(code, w) { acc += code * Math.pow(2, nb); nb += w; while (nb >= 8) { outb.push(acc % 256); acc = Math.floor(acc / 256); nb -= 8; } }
  let dict = new Map(), next = 258, width = 9;
  emit(256, width);
  let w = -1;
  for (let i = 0; i < data.length; i++) {
    const c = data[i];
    if (w < 0) { w = c; continue; }
    const key = w * 256 + c;
    if (dict.has(key)) { w = dict.get(key); continue; }
    emit(w, width);
    dict.set(key, next++);
    if (next > (1 << width) && width < 12) width++;
    if (next >= 4096) { emit(256, width); dict = new Map(); next = 258; width = 9; }
    w = c;
  }
  if (w >= 0) emit(w, width);
  emit(257, width);
  if (nb > 0) outb.push(acc % 256);
  let s = 0; for (let i = 0; i < data.length; i++) s = (s * 7 + data[i]) & 255;
  const head = Buffer.alloc(8); head.write('LZW1', 0, 'latin1'); head.writeUInt32BE(data.length, 4);
  out(Buffer.concat([head, Buffer.from(outb), Buffer.from([s])]));
}
function cmdUnpack(args) {
  const buf = readInput(args[0], 'unpack');
  if (buf.length < 9 || buf.toString('latin1', 0, 4) !== 'LZW1') fail('unpack: bad header');
  const total = buf.readUInt32BE(4), sum = buf[buf.length - 1], body = buf.subarray(8, buf.length - 1);
  let bp = 0;
  function read(w) {
    if (bp + w > body.length * 8) return -1;
    let v = 0;
    for (let i = 0; i < w; i++) { v += ((body[(bp + i) >> 3] >> ((bp + i) & 7)) & 1) * Math.pow(2, i); }
    bp += w; return v;
  }
  let tab = [], next = 258, width = 9, prev = null;
  const res = [];
  function expand(code) {
    const st = [];
    while (code >= 258) { const e = tab[code - 258]; st.push(e[1]); code = e[0]; }
    st.push(code); st.reverse(); return st;
  }
  while (true) {
    const code = read(width);
    if (code < 0) fail('unpack: truncated');
    if (code === 257) break;
    if (code === 256) { tab = []; next = 258; width = 9; prev = null; continue; }
    let seq;
    if (prev === null) { if (code > 255) fail('unpack: corrupt stream'); seq = [code]; }
    else if (code < next) seq = expand(code);
    else if (code === next) { const p = expand(prev); seq = p.concat([p[0]]); }
    else fail('unpack: corrupt stream');
    if (prev !== null) { tab.push([prev, seq[0]]); next++; if (next + 1 > (1 << width) && width < 12) width++; }
    for (const x of seq) res.push(x);
    if (res.length > total) fail('unpack: length mismatch');
    prev = code;
  }
  if (res.length !== total) fail('unpack: length mismatch');
  let s = 0; for (const x of res) s = (s * 7 + x) & 255;
  if (s !== sum) fail('unpack: crc mismatch');
  out(Buffer.from(res));
}

// ---------------------------------------------------------------- seal / open
function fnv32(bytes, seed) { let h = seed >>> 0; for (const b of bytes) { h = (h ^ b) >>> 0; h = Math.imul(h, 0x01000193) >>> 0; } return h; }
function rotl(x, k) { return ((x << k) | (x >>> (32 - k))) >>> 0; }
function keystream(key, nonce) {
  const kb = Buffer.from(key, 'utf8');
  const s = [0x811c9dc5, 0x1b873593, 0xcc9e2d51, 0x85ebca6b].map(sd => fnv32(kb, sd));
  s[0] = (s[0] ^ nonce) >>> 0;
  if ((s[0] | s[1] | s[2] | s[3]) === 0) s[0] = 1;
  return function () {
    const r = Math.imul(rotl(Math.imul(s[1], 5) >>> 0, 7), 9) >>> 0;
    const t = (s[1] << 9) >>> 0;
    s[2] = (s[2] ^ s[0]) >>> 0; s[3] = (s[3] ^ s[1]) >>> 0; s[1] = (s[1] ^ s[2]) >>> 0; s[0] = (s[0] ^ s[3]) >>> 0;
    s[2] = (s[2] ^ t) >>> 0; s[3] = rotl(s[3], 11);
    return r >>> 24;
  };
}
function macOf(key, data) { return crcOf(Buffer.concat([Buffer.from(key, 'utf8'), Buffer.from([0]), data])); }
function cmdSeal(args) {
  if (args.length < 1) fail('usage: tool seal KEY [FILE]', 2);
  const data = readInput(args[1], 'seal');
  const nonce = fnv32(Buffer.from([data.length & 255, (data.length >> 8) & 255, (data.length >> 16) & 255, (data.length >>> 24) & 255]), fnv32(Buffer.from(args[0], 'utf8'), 0x9747b28c));
  const ks = keystream(args[0], nonce);
  const c = Buffer.alloc(data.length);
  for (let i = 0; i < data.length; i++) c[i] = data[i] ^ ks();
  const nb = Buffer.alloc(4); nb.writeUInt32BE(nonce);
  const mb = Buffer.alloc(4); mb.writeUInt32BE(macOf(args[0], data));
  out(Buffer.concat([nb, c, mb]).toString('hex') + '\n');
}
function cmdOpen(args) {
  if (args.length < 1) fail('usage: tool open KEY [FILE]', 2);
  const text = readInput(args[1], 'open').toString('utf8').trim();
  if (text.length % 2 !== 0 || !/^[0-9a-fA-F]*$/.test(text)) fail('open: bad hex');
  const raw = Buffer.from(text, 'hex');
  if (raw.length < 8) fail('open: too short');
  const nonce = raw.readUInt32BE(0), c = raw.subarray(4, raw.length - 4), mac = raw.readUInt32BE(raw.length - 4);
  const ks = keystream(args[0], nonce);
  const p = Buffer.alloc(c.length);
  for (let i = 0; i < c.length; i++) p[i] = c[i] ^ ks();
  if (macOf(args[0], p) !== mac) fail('open: bad mac');
  out(p);
}

// ---------------------------------------------------------------- main
const CMDS = { crc: cmdCrc, b32: cmdB32, ser: cmdSer, de: cmdDe, expr: cmdExpr, tpl: cmdTpl, pack: cmdPack, unpack: cmdUnpack, seal: cmdSeal, open: cmdOpen };
(function main() {
  const a = process.argv.slice(2);
  if (a.length === 0) fail('usage: tool <crc|b32|ser|de|expr|tpl|pack|unpack|seal|open> ...', 2);
  if (a[0] === '--version') { out('tool 3.1.4\n'); return; }
  if (!Object.prototype.hasOwnProperty.call(CMDS, a[0])) fail("tool: unknown command '" + a[0] + "'", 2);
  CMDS[a[0]](a.slice(1));
})();
