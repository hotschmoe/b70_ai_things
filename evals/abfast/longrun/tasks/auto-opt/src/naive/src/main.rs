// nnjoin: for every document find the most similar OTHER document (Jaccard similarity of token sets).
//
// Input  (file given as argv[1]): first line N, then N lines; line i holds the token ids (decimal u32,
//         no leading zeros) of document i separated by spaces; a line may be empty; tokens may repeat
//         (a document is the SET of its tokens). Missing trailing lines count as empty documents.
// Output (stdout): N lines "J INTER UNION" for i = 0..N-1:
//         J     = index of the most similar other document (never i). Similarity of (i,j) is
//                 INTER/UNION as an exact fraction (INTER = |A and B|, UNION = |A or B|); a pair with
//                 UNION = 0 (both empty) has similarity 0. Ties are won by the LOWEST index j.
//         INTER, UNION = the intersection and union sizes of the chosen pair (UNION may be 0).
//         If N = 1 there is no other document: print "-1 0 0".
use std::env;
use std::fs;
use std::io::Write;

fn main() {
    let path = env::args().nth(1).expect("usage: nnjoin INPUT");
    let text = fs::read_to_string(&path).expect("cannot read input");
    let mut lines = text.lines();
    let n: usize = lines.next().unwrap_or("0").trim().parse().unwrap_or(0);
    let mut docs: Vec<Vec<String>> = Vec::new();
    for _ in 0..n {
        let line = lines.next().unwrap_or("");
        let mut d: Vec<String> = Vec::new();
        for tok in line.split_whitespace() {
            let s = tok.to_string();
            if !d.contains(&s) {
                d.push(s);
            }
        }
        docs.push(d);
    }
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    for i in 0..n {
        let mut best: Option<(usize, usize, usize)> = None;
        for j in 0..n {
            if j == i {
                continue;
            }
            let mut inter = 0usize;
            for a in &docs[i] {
                if docs[j].contains(a) {
                    inter += 1;
                }
            }
            let union = docs[i].len() + docs[j].len() - inter;
            match best {
                None => best = Some((j, inter, union)),
                Some((_, bi, bu)) => {
                    let (x1, y1) = if union == 0 { (0, 1) } else { (inter, union) };
                    let (x2, y2) = if bu == 0 { (0, 1) } else { (bi, bu) };
                    if x1 * y2 > x2 * y1 {
                        best = Some((j, inter, union));
                    }
                }
            }
        }
        match best {
            None => writeln!(out, "-1 0 0").unwrap(),
            Some((j, a, b)) => writeln!(out, "{} {} {}", j, a, b).unwrap(),
        }
    }
}
