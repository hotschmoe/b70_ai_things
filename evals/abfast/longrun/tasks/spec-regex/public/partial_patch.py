# applied to the reference main.rs to build the partial reference (match command only, no lazy/possessive/lookaround/backref/flags)
ANCHOR = "        let pat = l[0].as_bytes();\n"
INSERT = ANCHOR + '''        if !partial_ok(&a[1], l[0], l[1]) { writeln!(out, "E group").unwrap(); continue; }
'''
FUNC = '''
fn partial_ok(cmd: &str, pat: &str, flags: &str) -> bool {
    if cmd != "match" || !flags.is_empty() { return false; }
    for bad in ["(?=", "(?!", "(?<", "(?>", "(?P=", "(?i", "(?s", "(?m", "(?x", "(?-", "*?", "+?", "??", "}?", "*+", "++", "?+", "}+"] {
        if pat.contains(bad) { return false; }
    }
    let b = pat.as_bytes();
    for k in 0..b.len().saturating_sub(1) { if b[k] == b'\\\\' && b[k + 1] >= b'1' && b[k + 1] <= b'9' { return false; } }
    true
}
'''
