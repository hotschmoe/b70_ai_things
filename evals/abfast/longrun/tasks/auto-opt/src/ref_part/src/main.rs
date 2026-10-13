// partial reference: integer tokens + sorted merge, still O(N^2), single thread
use std::env;
use std::fs;
use std::io::Write;
fn main() {
    let path = env::args().nth(1).unwrap();
    let text = fs::read_to_string(&path).unwrap();
    let mut lines = text.lines();
    let n: usize = lines.next().unwrap_or("0").trim().parse().unwrap_or(0);
    let mut docs: Vec<std::collections::BTreeSet<u32>> = Vec::new();
    for _ in 0..n {
        let mut d: Vec<u32> = lines.next().unwrap_or("").split_whitespace().map(|t| t.parse().unwrap()).collect();
        docs.push(d.into_iter().collect());
    }
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    for i in 0..n {
        let mut best: Option<(usize, u64, u64)> = None;
        let a = &docs[i];
        for j in 0..n {
            if j == i { continue; }
            let b = &docs[j];
            let inter = a.iter().filter(|x| b.contains(x)).count() as u64;
            let union = (a.len() + b.len()) as u64 - inter;
            let better = match best {
                None => true,
                Some((_, bi, bu)) => {
                    let (x1, y1) = if union == 0 { (0, 1) } else { (inter, union) };
                    let (x2, y2) = if bu == 0 { (0, 1) } else { (bi, bu) };
                    x1 * y2 > x2 * y1
                }
            };
            if better { best = Some((j, inter, union)); }
        }
        match best {
            None => writeln!(out, "-1 0 0").unwrap(),
            Some((j, a, b)) => writeln!(out, "{} {} {}", j, a, b).unwrap(),
        }
    }
}
