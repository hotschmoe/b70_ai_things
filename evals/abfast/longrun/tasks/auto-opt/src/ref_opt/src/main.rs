// strong reference: dense token ids, inverted index with per-thread count arrays, std threads
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
#[derive(Default)]
struct Mul(u64);
impl Hasher for Mul {
    fn finish(&self) -> u64 { self.0 }
    fn write(&mut self, b: &[u8]) { for &x in b { self.0 = (self.0 ^ x as u64).wrapping_mul(0x9E3779B97F4A7C15); } }
    fn write_u32(&mut self, v: u32) { self.0 = (v as u64).wrapping_mul(0x9E3779B97F4A7C15) >> 7; }
}
use std::env;
use std::fs;
use std::io::Write;
fn main() {
    let path = env::args().nth(1).unwrap();
    let text = fs::read_to_string(&path).unwrap();
    let mut lines = text.lines();
    let n: usize = lines.next().unwrap_or("0").trim().parse().unwrap_or(0);
    let mut ids: HashMap<u32, u32, BuildHasherDefault<Mul>> = HashMap::default();
    let mut docs: Vec<Vec<u32>> = Vec::with_capacity(n);
    for _ in 0..n {
        let mut d: Vec<u32> = Vec::new();
        for t in lines.next().unwrap_or("").split_whitespace() {
            let nid = ids.len() as u32;
            let tv: u32 = t.parse().unwrap();
            d.push(*ids.entry(tv).or_insert(nid));
        }
        d.sort_unstable();
        d.dedup();
        docs.push(d);
    }
    let v = ids.len();
    let mut post: Vec<Vec<u32>> = vec![Vec::new(); v];
    for (i, d) in docs.iter().enumerate() {
        for &t in d { post[t as usize].push(i as u32); }
    }
    let mut res = vec![(-1i64, 0u64, 0u64); n];
    let nt = std::thread::available_parallelism().map(|x| x.get()).unwrap_or(1).min(8);
    let chunk = (n + nt - 1) / nt.max(1);
    std::thread::scope(|s| {
        for (ci, rc) in res.chunks_mut(chunk.max(1)).enumerate() {
            let docs = &docs;
            let post = &post;
            s.spawn(move || {
                let mut cnt = vec![0u32; n];
                let mut touched: Vec<u32> = Vec::new();
                for (k, slot) in rc.iter_mut().enumerate() {
                    let i = ci * chunk + k;
                    for &t in &docs[i] {
                        for &j in &post[t as usize] {
                            if cnt[j as usize] == 0 { touched.push(j); }
                            cnt[j as usize] += 1;
                        }
                    }
                    let la = docs[i].len() as u64;
                    let mut best: Option<(usize, u64, u64)> = None;
                    for &j in &touched {
                        let j = j as usize;
                        if j == i { continue; }
                        let inter = cnt[j] as u64;
                        let union = la + docs[j].len() as u64 - inter;
                        let better = match best {
                            None => true,
                            Some((bj, bi, bu)) => {
                                let l = inter * bu;
                                let r = bi * union;
                                l > r || (l == r && j < bj)
                            }
                        };
                        if better { best = Some((j, inter, union)); }
                    }
                    for &j in &touched { cnt[j as usize] = 0; }
                    touched.clear();
                    // inter == 0 everywhere: similarity 0, lowest index other than i wins
                    let zero_j = if i == 0 { 1 } else { 0 };
                    if n > 1 {
                        let take_zero = match best { None => true, Some(_) => false };
                        if take_zero {
                            let u = la + docs[zero_j].len() as u64;
                            *slot = (zero_j as i64, 0, u);
                        } else {
                            let (j, a, b) = best.unwrap();
                            *slot = (j as i64, a, b);
                        }
                    }
                }
            });
        }
    });
    let mut o = String::with_capacity(n * 16);
    for (j, a, b) in res { o.push_str(&format!("{} {} {}\n", j, a, b)); }
    std::io::stdout().write_all(o.as_bytes()).unwrap();
}
