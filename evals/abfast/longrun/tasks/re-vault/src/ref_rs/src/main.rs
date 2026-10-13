use std::fs;
use std::io::Write;
use std::os::unix::fs::PermissionsExt;
use std::process::exit;

const HDR: usize = 28;
const FNV_OFF: u64 = 0xcbf29ce484222325;

fn crc_tab() -> [u32; 256] {
    let mut t = [0u32; 256];
    for i in 0..256u32 {
        let mut c = i;
        for _ in 0..8 {
            c = if c & 1 != 0 { (c >> 1) ^ 0x82F63B78 } else { c >> 1 };
        }
        t[i as usize] = c;
    }
    t
}
fn vcrc(t: &[u32; 256], p: &[u8]) -> u32 {
    let mut c = 0xA5A5A5A5u32;
    for &b in p {
        c = t[((c ^ b as u32) & 0xff) as usize] ^ (c >> 8);
    }
    c ^ 0x3C3C3C3C
}
fn fnv64(p: &[u8], mut h: u64) -> u64 {
    for &b in p {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}
fn icheck(p: &[u8]) -> u32 {
    let mut h = 0x811C9DC5u32;
    for &b in p {
        h = h.rotate_left(5) ^ b as u32;
        h = h.wrapping_mul(0x01000193);
        h ^= h >> 15;
    }
    h
}
struct Ks(u64);
impl Ks {
    fn new(key: &str, salt: u64, idx: u32) -> Ks {
        let mut s = fnv64(key.as_bytes(), FNV_OFF) ^ salt ^ (idx as u64).wrapping_mul(0x9E3779B97F4A7C15);
        if s == 0 {
            s = 0x1234567;
        }
        Ks(s)
    }
    fn next(&mut self) -> u8 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        (x.wrapping_mul(0x2545F4914F6CDD1D) >> 56) as u8
    }
}
fn keycheck(key: &str, salt: u64) -> u32 {
    let h = fnv64(key.as_bytes(), FNV_OFF ^ salt);
    ((h ^ (h >> 32)) as u32) ^ 0xB16B00B5
}

fn rle_enc(inp: &[u8]) -> Vec<u8> {
    let n = inp.len();
    let (mut i, mut lit_start, mut lit) = (0usize, 0usize, 0usize);
    let mut out = Vec::new();
    while i < n {
        let mut r = 1;
        while i + r < n && inp[i + r] == inp[i] && r < 130 {
            r += 1;
        }
        if r >= 3 {
            if lit > 0 {
                out.push((lit - 1) as u8);
                out.extend_from_slice(&inp[lit_start..lit_start + lit]);
                lit = 0;
            }
            out.push(0x80 | (r - 3) as u8);
            out.push(inp[i]);
            i += r;
        } else {
            if lit == 0 {
                lit_start = i;
            }
            lit += 1;
            i += 1;
            if lit == 128 {
                out.push(127);
                out.extend_from_slice(&inp[lit_start..lit_start + lit]);
                lit = 0;
            }
        }
    }
    if lit > 0 {
        out.push((lit - 1) as u8);
        out.extend_from_slice(&inp[lit_start..lit_start + lit]);
    }
    out
}
fn rle_dec(inp: &[u8], rawn: usize) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(rawn);
    let mut i = 0;
    while i < inp.len() {
        let c = inp[i];
        i += 1;
        if c < 0x80 {
            let l = c as usize + 1;
            if i + l > inp.len() || out.len() + l > rawn {
                return None;
            }
            out.extend_from_slice(&inp[i..i + l]);
            i += l;
        } else {
            let l = (c & 0x7f) as usize + 3;
            if i >= inp.len() || out.len() + l > rawn {
                return None;
            }
            for _ in 0..l {
                out.push(inp[i]);
            }
            i += 1;
        }
    }
    if out.len() == rawn { Some(out) } else { None }
}
fn lz_enc(inp: &[u8]) -> Vec<u8> {
    let n = inp.len();
    let mut out: Vec<u8> = Vec::new();
    let (mut i, mut flag_at, mut bit) = (0usize, 0usize, 8);
    while i < n {
        if bit == 8 {
            flag_at = out.len();
            out.push(0);
            bit = 0;
        }
        let (mut best, mut bd) = (0usize, 0usize);
        let lo = if i > 4095 { i - 4095 } else { 0 };
        let mut j = i;
        while j > lo {
            j -= 1;
            let mut l = 0;
            while l < 18 && i + l < n && inp[j + l] == inp[i + l] {
                l += 1;
            }
            if l > best {
                best = l;
                bd = i - j;
            }
            if best == 18 {
                break;
            }
        }
        if best == 3 && bd > 2048 {
            best = 0;
        }
        if best >= 3 {
            out.push((bd & 0xff) as u8);
            out.push((((bd >> 8) << 4) | (best - 3)) as u8);
            i += best;
        } else {
            out[flag_at] |= 1 << bit;
            out.push(inp[i]);
            i += 1;
        }
        bit += 1;
    }
    out
}
fn lz_dec(inp: &[u8], rawn: usize) -> Option<Vec<u8>> {
    let mut out: Vec<u8> = Vec::with_capacity(rawn);
    let mut i = 0;
    while out.len() < rawn {
        if i >= inp.len() {
            return None;
        }
        let fl = inp[i];
        i += 1;
        let mut b = 0;
        while b < 8 && out.len() < rawn {
            if fl & (1 << b) != 0 {
                if i >= inp.len() {
                    return None;
                }
                out.push(inp[i]);
                i += 1;
            } else {
                if i + 2 > inp.len() {
                    return None;
                }
                let d = inp[i] as usize | (((inp[i + 1] >> 4) as usize) << 8);
                let l = (inp[i + 1] & 15) as usize + 3;
                i += 2;
                if d == 0 || d > out.len() || out.len() + l > rawn {
                    return None;
                }
                for _ in 0..l {
                    let v = out[out.len() - d];
                    out.push(v);
                }
            }
            b += 1;
        }
    }
    Some(out)
}

struct Ent {
    name: String,
    method: u8,
    mode: u16,
    raw: u32,
    comp: u32,
    off: u32,
    crc: u32,
}
struct Arc {
    buf: Vec<u8>,
    flags: u16,
    ioff: u32,
    salt: u64,
    kc: u32,
    e: Vec<Ent>,
    idx_ok: bool,
}
fn rd16(p: &[u8]) -> u16 { u16::from_le_bytes([p[0], p[1]]) }
fn rd32(p: &[u8]) -> u32 { u32::from_le_bytes([p[0], p[1], p[2], p[3]]) }
fn rd64(p: &[u8]) -> u64 { rd32(p) as u64 | ((rd32(&p[4..]) as u64) << 32) }

fn die(code: i32, msg: String) -> ! {
    eprintln!("vault: {}", msg);
    exit(code)
}
fn unsafe_name(n: &str) -> bool {
    if n.is_empty() || n.starts_with('/') {
        return true;
    }
    for part in n.split('/') {
        if part == ".." || part.is_empty() {
            return true;
        }
    }
    false
}
fn arc_open(path: &str) -> Arc {
    let buf = match fs::read(path) {
        Ok(b) => b,
        Err(_) => die(6, format!("cannot read {}", path)),
    };
    if buf.len() < HDR || &buf[0..4] != b"VLT3" {
        die(3, format!("{}: not a vault archive", path));
    }
    if rd16(&buf[4..]) != 3 {
        die(3, format!("{}: unsupported version", path));
    }
    let flags = rd16(&buf[6..]);
    let count = rd32(&buf[8..]);
    let ioff = rd32(&buf[12..]);
    let salt = rd64(&buf[16..]);
    let kc = rd32(&buf[24..]);
    if (ioff as usize) < HDR || ioff as usize > buf.len() {
        die(3, format!("{}: corrupt index offset", path));
    }
    let mut p = ioff as usize;
    let end = buf.len();
    let mut e = Vec::new();
    for _ in 0..count {
        if p + 2 > end {
            die(3, format!("{}: truncated index", path));
        }
        let nl = rd16(&buf[p..]) as usize;
        p += 2;
        if p + nl + 19 > end {
            die(3, format!("{}: truncated index", path));
        }
        let name: String = buf[p..p + nl].iter().map(|&b| b as char).collect();
        p += nl;
        let method = buf[p];
        let mode = rd16(&buf[p + 1..]);
        let raw = rd32(&buf[p + 3..]);
        let comp = rd32(&buf[p + 7..]);
        let off = rd32(&buf[p + 11..]);
        let crc = rd32(&buf[p + 15..]);
        p += 19;
        if off as u64 + comp as u64 > ioff as u64 || (off as usize) < HDR {
            die(3, format!("{}: entry out of range", path));
        }
        e.push(Ent { name, method, mode, raw, comp, off, crc });
    }
    if p + 4 > end {
        die(3, format!("{}: truncated index", path));
    }
    let idx_ok = icheck(&buf[ioff as usize..p]) == rd32(&buf[p..]);
    Arc { buf, flags, ioff, salt, kc, e, idx_ok }
}
fn ent_data(a: &Arc, i: usize, key: Option<&str>) -> Result<Vec<u8>, i32> {
    let e = &a.e[i];
    let mut c = a.buf[e.off as usize..(e.off + e.comp) as usize].to_vec();
    if a.flags & 1 != 0 {
        let key = key.ok_or(1)?;
        let mut ks = Ks::new(key, a.salt, i as u32);
        for b in c.iter_mut() {
            *b ^= ks.next();
        }
    }
    let r = match e.method {
        0 => if e.comp == e.raw { Some(c) } else { None },
        1 => rle_dec(&c, e.raw as usize),
        2 => lz_dec(&c, e.raw as usize),
        _ => None,
    };
    r.ok_or(2)
}
fn check_key(a: &Arc, key: Option<&str>) {
    if a.flags & 1 == 0 {
        return;
    }
    match key {
        None => die(4, "archive is encrypted; key required".into()),
        Some(k) => if keycheck(k, a.salt) != a.kc { die(4, "wrong key".into()) },
    }
}

fn take_key(args: &mut Vec<String>) -> Option<String> {
    if args.len() >= 2 && args[0] == "-k" {
        let k = args[1].clone();
        args.drain(0..2);
        Some(k)
    } else {
        None
    }
}

fn main() {
    let t = crc_tab();
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        die(2, "usage: vault pack|list|info|cat|extract|verify|digest ...".into());
    }
    let cmd = args.remove(0);
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    match cmd.as_str() {
        "digest" => {
            if args.len() != 1 { die(2, "usage: vault digest FILE".into()); }
            let b = fs::read(&args[0]).unwrap_or_else(|_| die(6, format!("cannot read {}", args[0])));
            writeln!(out, "crc={:08x} fnv64={:016x} len={}", vcrc(&t, &b), fnv64(&b, FNV_OFF), b.len()).unwrap();
        }
        "info" => {
            if args.len() != 1 { die(2, "usage: vault info ARCHIVE".into()); }
            let a = arc_open(&args[0]);
            let raw: u64 = a.e.iter().map(|e| e.raw as u64).sum();
            let comp: u64 = a.e.iter().map(|e| e.comp as u64).sum();
            write!(out, "format: VLT3 v3\nentries: {}\nencrypted: {}\nindex offset: {}\nindex check: {}\nraw bytes: {}\npacked bytes: {}\n",
                a.e.len(), if a.flags & 1 != 0 { "yes" } else { "no" }, a.ioff, if a.idx_ok { "OK" } else { "BAD" }, raw, comp).unwrap();
        }
        "list" => {
            let lng = !args.is_empty() && args[0] == "-l";
            if lng { args.remove(0); }
            if args.len() != 1 { die(2, "usage: vault list [-l] ARCHIVE".into()); }
            let a = arc_open(&args[0]);
            let (mut raw, mut comp) = (0u64, 0u64);
            for e in &a.e {
                raw += e.raw as u64;
                comp += e.comp as u64;
                if lng {
                    let mn = ["store", "rle", "lz"];
                    writeln!(out, "{:04o} {:<5} {:>8} {:>8} {:08x} {}", e.mode & 0o777,
                        if (e.method as usize) < 3 { mn[e.method as usize] } else { "?" }, e.raw, e.comp, e.crc, e.name).unwrap();
                } else {
                    writeln!(out, "{}", e.name).unwrap();
                }
            }
            if lng { writeln!(out, "{} entries, {} bytes ({} packed)", a.e.len(), raw, comp).unwrap(); }
        }
        "cat" => {
            let key = take_key(&mut args);
            if args.len() != 2 { die(2, "usage: vault cat [-k KEY] ARCHIVE NAME".into()); }
            let a = arc_open(&args[0]);
            check_key(&a, key.as_deref());
            for i in 0..a.e.len() {
                if a.e[i].name == args[1] {
                    let d = ent_data(&a, i, key.as_deref()).unwrap_or_else(|_| die(3, format!("{}: corrupt entry", args[1])));
                    if vcrc(&t, &d) != a.e[i].crc { die(3, format!("{}: checksum mismatch", args[1])); }
                    out.write_all(&d).unwrap();
                    return;
                }
            }
            die(5, format!("{}: no such entry", args[1]));
        }
        "extract" => {
            let key = take_key(&mut args);
            if args.len() < 2 { die(2, "usage: vault extract [-k KEY] ARCHIVE DIR [NAME...]".into()); }
            let a = arc_open(&args[0]);
            check_key(&a, key.as_deref());
            for n in &args[2..] {
                if !a.e.iter().any(|e| &e.name == n) { die(5, format!("{}: no such entry", n)); }
            }
            for i in 0..a.e.len() {
                let e = &a.e[i];
                if args.len() > 2 && !args[2..].contains(&e.name) { continue; }
                if unsafe_name(&e.name) { die(3, format!("{}: unsafe entry name", e.name)); }
                let d = ent_data(&a, i, key.as_deref()).unwrap_or_else(|_| die(3, format!("{}: corrupt entry", e.name)));
                if vcrc(&t, &d) != e.crc { die(3, format!("{}: checksum mismatch", e.name)); }
                let path = format!("{}/{}", args[1], e.name);
                if let Some(pos) = path.rfind('/') { let _ = fs::create_dir_all(&path[..pos]); }
                fs::write(&path, &d).unwrap_or_else(|_| die(6, format!("cannot write {}", path)));
                let _ = fs::set_permissions(&path, fs::Permissions::from_mode((e.mode & 0o777) as u32));
                writeln!(out, "x {}", e.name).unwrap();
            }
        }
        "verify" => {
            let key = take_key(&mut args);
            if args.len() != 1 { die(2, "usage: vault verify [-k KEY] ARCHIVE".into()); }
            let a = arc_open(&args[0]);
            check_key(&a, key.as_deref());
            let mut bad = 0;
            if !a.idx_ok { writeln!(out, "BAD index (checksum mismatch)").unwrap(); bad += 1; }
            for i in 0..a.e.len() {
                match ent_data(&a, i, key.as_deref()) {
                    Err(_) => { writeln!(out, "BAD {} (corrupt data)", a.e[i].name).unwrap(); bad += 1; }
                    Ok(d) => if vcrc(&t, &d) != a.e[i].crc { writeln!(out, "BAD {} (crc mismatch)", a.e[i].name).unwrap(); bad += 1; }
                        else { writeln!(out, "OK {}", a.e[i].name).unwrap(); }
                }
            }
            if bad > 0 {
                writeln!(out, "FAILED {} problem{}", bad, if bad == 1 { "" } else { "s" }).unwrap();
                out.flush().unwrap();
                exit(1);
            }
            writeln!(out, "verified {} entries", a.e.len()).unwrap();
        }
        "pack" => {
            let mut key: Option<String> = None;
            let mut method: i32 = -1;
            while args.len() >= 2 && args[0].starts_with('-') {
                match args[0].as_str() {
                    "-k" => key = Some(args[1].clone()),
                    "-m" => match args[1].as_str() {
                        "auto" => method = -1,
                        "0" => method = 0,
                        "1" => method = 1,
                        "2" => method = 2,
                        m => die(2, format!("bad method {}", m)),
                    },
                    o => die(2, format!("unknown option {}", o)),
                }
                args.drain(0..2);
            }
            if args.len() < 2 { die(2, "usage: vault pack [-k KEY] [-m 0|1|2|auto] OUT FILE...".into()); }
            let outp = args[0].clone();
            let files: Vec<String> = args[1..].to_vec();
            let mut names: Vec<String> = Vec::new();
            for f in &files {
                let mut n = f.as_str();
                while n.starts_with("./") { n = &n[2..]; }
                if unsafe_name(n) { die(3, format!("{}: unsafe path", f)); }
                if names.iter().any(|x| x == n) { die(2, format!("{}: duplicate entry", n)); }
                names.push(n.to_string());
            }
            let nf = files.len();
            let mut salt = 0u64;
            if key.is_some() {
                salt = FNV_OFF;
                for n in &names { salt = fnv64(n.as_bytes(), salt); }
                salt ^= nf as u64;
            }
            let mut buf = vec![0u8; HDR];
            let mut es: Vec<Ent> = Vec::new();
            for i in 0..nf {
                let d = fs::read(&files[i]).unwrap_or_else(|_| die(6, format!("cannot read {}", files[i])));
                let mode = fs::metadata(&files[i]).map(|m| m.permissions().mode() & 0o777).unwrap_or(0) as u16;
                let (mut best, mut bm) = (d.clone(), 0u8);
                if method != 0 {
                    for m in 1..=2 {
                        if method >= 0 && m != method { continue; }
                        let tmp = if m == 1 { rle_enc(&d) } else { lz_enc(&d) };
                        if method >= 0 || tmp.len() < best.len() { best = tmp; bm = m as u8; }
                    }
                }
                let crc = vcrc(&t, &d);
                let off = buf.len() as u32;
                let comp = best.len() as u32;
                if let Some(k) = &key {
                    let mut ks = Ks::new(k, salt, i as u32);
                    for b in best.iter_mut() { *b ^= ks.next(); }
                }
                buf.extend_from_slice(&best);
                es.push(Ent { name: names[i].clone(), method: bm, mode, raw: d.len() as u32, comp, off, crc });
            }
            let ioff = buf.len();
            for e in &es {
                buf.extend_from_slice(&(e.name.len() as u16).to_le_bytes());
                buf.extend_from_slice(e.name.as_bytes());
                buf.push(e.method);
                buf.extend_from_slice(&e.mode.to_le_bytes());
                for v in [e.raw, e.comp, e.off, e.crc] { buf.extend_from_slice(&v.to_le_bytes()); }
            }
            let ic = icheck(&buf[ioff..]);
            buf.extend_from_slice(&ic.to_le_bytes());
            buf[0..4].copy_from_slice(b"VLT3");
            buf[4..6].copy_from_slice(&3u16.to_le_bytes());
            buf[6..8].copy_from_slice(&(if key.is_some() { 1u16 } else { 0 }).to_le_bytes());
            buf[8..12].copy_from_slice(&(nf as u32).to_le_bytes());
            buf[12..16].copy_from_slice(&(ioff as u32).to_le_bytes());
            buf[16..24].copy_from_slice(&salt.to_le_bytes());
            let kc = key.as_ref().map(|k| keycheck(k, salt)).unwrap_or(0);
            buf[24..28].copy_from_slice(&kc.to_le_bytes());
            fs::write(&outp, &buf).unwrap_or_else(|_| die(6, format!("cannot write {}", outp)));
            writeln!(out, "packed {} entr{}, {} bytes", nf, if nf == 1 { "y" } else { "ies" }, buf.len()).unwrap();
        }
        c => die(2, format!("unknown command {}", c)),
    }
}
