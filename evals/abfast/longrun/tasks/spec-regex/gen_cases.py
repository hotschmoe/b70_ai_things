#!/usr/bin/env python3
"""Record hidden + public cases for spec-regex.  gen_cases.py HIDDEN_DIR PUBLIC_DIR RX_REF_BINARY"""
import json, os, random, subprocess, sys, tempfile
H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(H, "src"))
import oracle_re as O

HID, PUB, RXBIN = sys.argv[1], sys.argv[2], sys.argv[3]
cases = []


def case(ms, cmd, pat, flags, subj, tmpl=None, expect=None, timeout=10):
    assert "\n" not in pat and (tmpl is None or "\n" not in tmpl)
    lines = [pat, flags] + ([tmpl] if cmd == "sub" else []) + [O.enc(subj)]
    if expect is None:
        expect = {"match": lambda: O.match_line(pat, flags, subj), "findall": lambda: O.findall_line(pat, flags, subj),
                  "sub": lambda: O.sub_line(pat, flags, tmpl, subj)}[cmd]()
    cases.append({"id": f"{ms}-{len([c for c in cases if c['ms'] == ms]):03d}", "ms": ms, "cmd": cmd, "input": "\n".join(lines) + "\n",
                  "expect": expect, "timeout": timeout})


def M(ms, rows, cmd="match"):
    for r in rows:
        case(ms, cmd, r[0], r[1], r[2])


M("r01", [("abc", "", "xxabcxx"), ("abc", "", "ab"), ("a.c", "", "abc"), ("a.c", "", "a\nc"), ("a\\.c", "", "abc"), ("a\\.c", "", "a.c"), ("\\n", "", "x\ny"),
          ("\\t", "", "a\tb"), ("\\x41", "", "zAz"), ("a\\x2eb", "", "a.b"), ("\\(\\)", "", "f()"), ("\\\\", "", "a\\b"), ("\\*\\+\\?", "", "*+?"), ("a{", "", "a{"),
          ("a{b", "", "xa{b"), ("{", "", "{"), ("}", "", "a}"), ("\\/", "", "a/b"), ("", "", "abc"), ("", "", ""), ("abc", "", ""), ("\\-", "", "a-b"), ("\\ ", "", "a b"),
          ("\\v", "", "a\x0bb"), ("\\f", "", "\x0c"), ("\\r\\n", "", "a\r\nb"), ("...", "", "ab"), ("a.b.c", "", "a-b_c"), ("\\x7f", "", "\x7f"), ("\\x00", "", "a\x00b"),
          ("a#b", "", "a#b"), ("a b", "", "a b"), ("\\#", "", "#"), ("\\^\\$", "", "^$")])
M("r02", [("^abc", "", "abcabc"), ("^b", "", "abc"), ("abc$", "", "xabc"), ("abc$", "", "abc\n"), ("abc\\Z", "", "abc\n"), ("\\Aabc", "", "abc"), ("^b", "m", "a\nb"),
          ("a$", "m", "a\nb"), ("^$", "", ""), ("^$", "m", "a\n\nb"), ("$", "", "abc"), ("$", "", "abc\n"), ("\\bfoo\\b", "", "a foo b"), ("\\bfoo\\b", "", "afoo b"),
          ("\\Boo", "", "foo"), ("\\b", "", ""), ("\\b", "", "  a"), ("^^a", "", "a"), ("a\\b", "", "a!"), ("\\B", "", "ab"), ("\\B", "", "a"), ("^", "m", "\n"), ("$", "m", "\n"),
          ("\\A", "m", "a"), ("\\Z", "m", "a\n"), ("a$\\n", "", "a\n"), ("^a$", "m", "b\na\nc"), ("\\bab", "", "_ab ab"), ("ab\\B", "", "abc ab"), ("^\\s*$", "m", "x\n  \ny"),
          ("(^|,)b", "", "a,b"), ("b($|,)", "", "a,b"), ("\\Bb\\B", "", "abc")])
M("r03", [("[abc]", "", "xxb"), ("[^abc]", "", "abcd"), ("[a-c]+", "", "xabcabcx"), ("[a-cx-z]+", "", "dxyz"), ("[\\d]+", "", "ab123"), ("[\\w.]+", "", "!a.b_c!"),
          ("[\\s]", "", "ab c"), ("[^\\s]+", "", "  hi "), ("[]a]+", "", "x]a]"), ("[^]a]", "", "]ab"), ("[a-]+", "", "x-a-"), ("[-a]+", "", "-a"), ("[a\\-z]+", "", "a-z"),
          ("[\\]]", "", "]"), ("[\\^a]+", "", "^a"), ("[.]", "", "a.b"), ("[*+?]+", "", "*+?"), ("\\d\\D\\w\\W\\s\\S", "", "1a_ \tZ"), ("[\\x41-\\x43]+", "", "zABCD"),
          ("[\\n]", "", "a\nb"), ("[a-zA-Z_][a-zA-Z0-9_]*", "", "  foo_bar9 "), ("[\\b]", "", "a\x08"), ("[^\\d\\s]+", "", "12 ab 34"), ("[a-a]", "", "a"), ("[[]", "", "["),
          ("[\\\\]", "", "a\\b"), ("[$^]", "", "a$"), ("[a-c]", "i", "B"), ("[^a-c]", "i", "B"), ("[^a-c]", "i", "d"), ("[^\\n]+", "", "ab\ncd"), ("[\\W\\d]+", "", "ab-1 c"),
          ("[a-z&&[^b]]", "", "b&"), ("[ -~]+", "", "\x01abc \x7f")])
M("r04", [("a|b", "", "xb"), ("abc|abd", "", "abd"), ("(a|ab)(c|bcd)(d*)", "", "abcd"), ("(a)(b)(c)", "", "abc"), ("(a(b(c)))", "", "abc"), ("(?:a|b)+", "", "abba"),
          ("(a)|(b)", "", "b"), ("(a)|(b)", "", "a"), ("x(a|b|c)y", "", "xcy"), ("(?P<first>a)(?P<second>b)", "", "ab"), ("(a|b)*", "", "abab"), ("(?:ab)+", "", "ababab"),
          ("()", "", "a"), ("a||b", "", "b"), ("(a)?b", "", "b"), ("((a)|(b))+", "", "ab"), ("(a+)(b+)?", "", "aac"), ("(a+)(b+)?", "", "aabb"), ("a|ab|abc", "", "abc"),
          ("(?:(a)|b)*", "", "ab"), ("((a)(b))|((c)(d))", "", "cd"), ("(a)(?:(b)|(c))", "", "ac"), ("(?:)", "", "a"), ("|a", "", "a"), ("a|", "", "b"), ("(a|b)(c|d)", "", "xbd")])
M("r05", [("a*", "", "aaab"), ("a+", "", "baaa"), ("a?", "", "aa"), ("a{2}", "", "aaaa"), ("a{2,}", "", "aaaa"), ("a{,2}", "", "aaaa"), ("a{1,3}", "", "aaaa"), ("a{0}", "", "aaa"),
          ("ba{2,3}", "", "baaaa"), ("(ab){2}", "", "ababab"), ("a*b", "", "aaab"), ("a*ab", "", "aaab"), (".*c", "", "abcabc"), (".+", "", "a\nb"), ("x*", "", "yyy"),
          ("[ab]{3}", "", "ababab"), ("(a|b){2,3}", "", "abab"), ("\\d{3}-\\d{4}", "", "call 555-1234 now"), ("a{1", "", "a{1"), ("a{1,", "", "a{1,"), ("a{x}", "", "a{x}"),
          ("a{2", "", "aa{2"), ("a{ 1}", "", "a{ 1}"), ("(a{2}){2}", "", "aaaaa"), ("a{0,0}b", "", "aab"), ("a{3}", "", "aa"), ("(\\d+)\\.(\\d+)", "", "pi 3.14!"),
          ("a{1000}", "", "a" * 1000), ("(?:a{10}){10}", "", "a" * 100), ("x{2,4}y", "", "xxxxxy"), ("\\w+@\\w+\\.com", "", "mail bob@site.com now")])
M("r06", [("a*?", "", "aaa"), ("a+?", "", "aaa"), ("a??", "", "a"), ("<.*?>", "", "<a><b>"), ("<.*>", "", "<a><b>"), ("a{2,3}?", "", "aaaa"), ("(a+?)(b*)", "", "aaab"),
          (".*?c", "", "abcabc"), ("(.*?),(.*)", "", "a,b,c"), ("a*?b", "", "aaab"), ("\\d+?\\d", "", "123"), ("(a|ab)*?c", "", "ababc"), ("a{2,}?", "", "aaaa"),
          ("(.+?)\\1", "", "abab"), ("\\s*?x", "", "  x")])
M("r07", [("(a)\\1", "", "aa"), ("(a)\\1", "", "ab"), ("(\\w)\\1", "", "hello"), ("(a)(b)\\2\\1", "", "abba"), ("(?P<x>a)(?P=x)", "", "aa"), ("(a)?\\1", "", "b"), ("(a)|b\\1", "", "b"),
          ("(['\"])(.*?)\\1", "", "say \"hi\" 'x'"), ("(a+)\\1", "", "aaaa"), ("(a*)b\\1", "", "aabaa"), ("(?i)(a)\\1", "", "aA"), ("(.)\\1\\1", "", "abbbc"), ("(a)(?:b\\1)+", "", "abab" + "a"),
          ("(\\d+)-\\1", "", "12-12 13-14"), ("(?P<q>\"|')\\w+(?P=q)", "", "'ab\" \"cd\"")])
M("r08", [("abc", "i", "xABCx"), ("[a-z]+", "i", "ABC"), ("a.c", "s", "a\nc"), ("a.c", "", "a\nc"), ("^b", "m", "a\nb"), ("a b", "x", "ab"), ("a b # comment", "x", "ab"), ("a\\ b", "x", "a b"),
          ("[ ]", "x", " "), ("a  b", "x", "a  b"), ("(?i)abc", "", "ABC"), ("(?s).", "", "\n"), ("(?m)^b", "", "a\nb"), ("(?x) a b", "", "ab"), ("(?is)A.B", "", "a\nb"),
          ("(?i:a)b", "", "Ab"), ("(?i:a)b", "", "AB"), ("a(?i:b)c", "", "aBc"), ("a(?i:b)c", "", "aBC"), ("(?-i:a)b", "i", "AB"), ("(?-i:a)b", "i", "aB"), ("(?s:.)", "", "\n"),
          ("(?x: a b )c", "", "abc"), ("(?m:^a)", "", "b\na"), ("a\\#b", "x", "a#b"), ("a # x\n".rstrip("\n"), "x", "a"), ("\\d+ \\s* x", "x", "12  x"), ("A", "i", "a"),
          ("ab+c", "ix", "xABBC"), ("a b c", "x", "abc"), ("a\\x20b", "x", "a b"), ("(?i)a(?-i:b)", "", "AB"), ("(?i)a(?-i:b)", "", "Ab"), ("a +", "x", "aaa"), ("(?i)[A-C]+", "", "abcd")])
M("r09", [("a(?=b)", "", "ab"), ("a(?=b)", "", "ac"), ("a(?!b)", "", "abac"), ("(?<=a)b", "", "ab"), ("(?<!a)b", "", "abcb"), ("(?<=ab|cd)e", "", "cde"), ("(?<=\\d\\d)x", "", "12x"),
          ("(?=(a+))a*b\\1", "", "baaabac"), ("(?<=\\bfoo)bar", "", "foobar"), ("\\w+(?=,)", "", "ab,cd"), ("(?=a)", "", "ba"), ("(?!)", "", "a"), ("(?<=a{2})b", "", "aab"),
          ("(?<=(a))b", "", "ab"), ("(?=(\\w+))\\1:", "", "abc:"), ("(?!a)\\w", "", "ab"), ("q(?!u)", "", "quit qat"), ("(?<=^)a", "", "a"), ("(?<!^)a", "", "aa"),
          ("(?<=\\s)\\w+", "", "ab cd"), ("\\d+(?=%)", "", "10 20%"), ("(?<![a-z])\\d", "", "a1 2"), ("(?<=a|b)c", "", "bc"), ("x(?=(?=y)y)", "", "xy"), ("(?<=(?<=a)b)c", "", "abc")])
M("r10", [("a++b", "", "aaab"), ("a++a", "", "aaa"), ("a*+a", "", "aaa"), ("a?+a", "", "a"), ("a{1,2}+a", "", "aaa"), ("(?>a+)b", "", "aab"), ("(?>a|ab)c", "", "abc"), ("(a|ab)c", "", "abc"),
          ("(?>a*)a", "", "aaa"), ("\"[^\"]*+\"", "", "say \"hi\""), ("[^,]*+,", "", "ab,cd"), ("x*+x", "", "xx"), ("(?>(a+))b", "", "aaab"), ("a+?+", "", "aa"), ("\\w++\\d", "", "ab12"),
          ("(?>\\d+)\\d", "", "123"), ("(?:a|b)++c", "", "abac"), ("\\s*+x", "", "  x")])
M("r11", [("(", "", "a"), (")", "", "a"), ("(a", "", "a"), ("a)", "", "a"), ("[a", "", "a"), ("[z-a]", "", "a"), ("*a", "", "a"), ("a**", "", "a"), ("a+*", "", "a"), ("a{3,2}", "", "a"),
          ("?", "", "a"), ("+", "", "a"), ("a|*", "", "a"), ("\\q", "", "a"), ("\\", "", "a"), ("\\x4", "", "a"), ("\\xZZ", "", "a"), ("\\1", "", "a"), ("(a)\\2", "", "a"), ("(a\\1)", "", "a"),
          ("(?P<n>a)(?P=m)", "", "a"), ("(?P<1>a)", "", "a"), ("(?P<n>a)(?P<n>b)", "", "a"), ("(?<=a+)b", "", "a"), ("(?<=a|bc)d", "", "a"), ("x(?i)a", "", "a"), ("(?z)", "", "a"),
          ("a{2}{3}", "", "a"), ("^*", "", "a"), ("(?P<a", "", "a"), ("[\\d-z]", "", "a"), ("(?:", "", "a"), ("(()", "", "a"), ("a|(", "", "a"), ("(?<=\\1)a", "", "a"),
          ("()\\2", "", "a"), ("\\8", "", "a"), ("(?P<n>", "", "a"), ("[\\d", "", ""), ("a\\Q", "", "a"), ("(?<n>a)", "", "a")])
M("r12", [("a", "", "banana"), ("a*", "", "baac"), ("\\d+", "", "a1b22c333"), ("", "", "abc"), ("x*", "", "axxb"), ("\\b", "", "ab cd"), ("^", "m", "a\nb\nc"), ("$", "m", "a\nb\n"),
          ("(?=a)", "", "aaa"), ("a|ab", "", "abab"), ("\\w+", "", "hello world"), ("[^,]*", "", "a,b,,c"), ("a??", "", "aa"), ("(a)|b", "", "ab"), ("", "", ""), ("a*?", "", "aa"),
          ("\\s*", "", "a  b"), ("b|", "", "abc"), ("(?:ab|a)*", "", "aab"), ("\\B", "", "ab c"), ("a+|b*", "", "aab"), ("$", "", "a\n"), ("^a", "", "aa"), ("^a", "m", "a\na"), ("x", "", "abc"),
          ("(?<=a)", "", "aa"), (".", "", "a\nb"), ("\\d*", "", "12ab3")], cmd="findall")
SUBS = [("a", "-", "banana"), ("(\\w+) (\\w+)", "\\2 \\1", "hello world"), ("a*", "-", "baac"), ("(?P<w>\\w+)", "<\\g<w>>", "ab cd"), ("(a)|b", "[\\1]", "ab"), ("\\d+", "\\g<0>\\g<0>", "a1b22"),
        ("x", "\\\\", "axb"), ("x", "\\n", "axb"), ("x", "\\t", "axb"), ("(a)(b)", "\\2\\1", "abab"), ("", "-", "abc"), ("$", "!", "abc"), ("a", "\\1", "a"), ("a", "\\q", "a"), ("a", "\\g<x>", "a"),
        ("(a)", "\\g<1>x", "a"), ("b*", "X", "abc"), ("\\s+", " ", "a  b \t c"), ("(\\d)(\\d)", "\\2\\1", "1234"), ("a", "", "banana"), ("a", "\\", "a"), ("(?P<x>a)", "\\g<x>\\g<x>", "xa"),
        ("^", "> ", "abc"), ("x", "[\\g<0>]", "axbx"), ("", "\\g<0>-", "ab"), ("(.)\\1", "<\\1>", "aabbc"), ("a|(b)", "(\\1)", "ab"), ("\\n", "\\\\n", "a\nb"), ("(a)", "\\2", "a"), ("a", "\\r", "a")]
for p, t, s in SUBS:
    fl = "m" if p == "^" else ""
    case("r13", "sub", p, fl, s, tmpl=t)
case("r13", "sub", "^", "m", "a\nb", tmpl="> ")
case("r13", "sub", "$", "m", "a\nb", tmpl="!")
M("r14", [("a*?a", "", "aaa"), ("(a*)(a*)", "", "aaa"), ("(a*?)(a*)", "", "aaa"), ("(a+)(a+)", "", "aaaa"), ("(.*)(\\d+)", "", "abc123"), ("(.*?)(\\d+)", "", "abc123"), ("((a)|b)*", "", "abab"),
          ("(a|b)+?c", "", "abbc"), ("(a)*b", "", "aab"), ("^(a+)+$", "", "aaaa"), ("(\\w+)\\s(\\w+)\\s(\\w+)", "", "one two three"), ("([a-z]+)(\\d*)", "", "abc123"), ("(x+x+)y", "", "xxxy"),
          ("(a|aa)+b", "", "aaab"), ("\\b(\\w+)\\s+\\1\\b", "", "the the cat"), ("^(?:(\\d+)|(\\w+))$", "", "abc"), ("(?:a{2})+", "", "aaaaa"), ("(a{1,2}){2}", "", "aaa"), ("a|b|c|d", "", "d"),
          ("(a?)*?b", "", "aab"), ("(ab|a)(bc|c)?", "", "abc"), ("(a*)*", "", "aab"), ("(a*)+", "", "b"), ("(a|b*)*", "", "ab"), ("(a?){3}", "", "aa"), ("((a)|(b))*c", "", "abac"),
          ("(?:(a)|(b))+", "", "ba"), ("(a+|b+)*c", "", "aabbc"), ("(\\d+)(?:\\.(\\d+))?", "", "v3.14 v2"), ("^(a|ab)(c|bcd)(d*)$", "", "abcd")])
# performance: expectations from the reference implementation; sanity-checked against python at small sizes in build.py
PERF = [("(a+)+$", "", "a" * 50 + "!"), ("(a|aa)+$", "", "a" * 60 + "!"), ("(x+x+)+y", "", "x" * 60), ("^(\\w+\\s?)*$", "", "word " * 14 + "!"), ("(a?){30}a{30}", "", "a" * 30),
        ("(.*a){12}", "", "a" * 40), ("\\w*\\w*\\w*\\w*\\w*x", "", "a" * 300), ("(\\d+)*\\.", "", "1" * 50), ("(a*a*)*b" if False else "(a+a+)+b", "", "a" * 45),
        ("^(?:a|a)*$", "", "a" * 30 + "b"), ("(?:(?:a+)+)+$", "", "a" * 40 + "b")]
PERF_SMALL = {}
for i, (p, f, s) in enumerate(PERF):
    cases.append({"id": f"r15-{i:03d}", "ms": "r15", "cmd": "match", "input": "\n".join([p, f, O.enc(s)]) + "\n", "expect": None, "timeout": 15})
# fill perf expectations with the reference binary; check against python on shrunken subjects
def run_rx(cmd, text, timeout=30):
    with tempfile.NamedTemporaryFile("w", delete=False) as fh:
        fh.write(text)
    try:
        return subprocess.run([RXBIN, cmd, fh.name], capture_output=True, text=True, timeout=timeout).stdout.rstrip("\n").split("\n")
    finally:
        os.unlink(fh.name)


for c in cases:
    if c["ms"] == "r15":
        c["expect"] = run_rx("match", c["input"])[0]
        p, f, s = c["input"].split("\n")[:3]
        subj = s
        small = subj[:9] + subj[-1:] if len(subj) > 11 else subj
        py = O.match_line(p, f, small)
        got = run_rx("match", f"{p}\n{f}\n{small}\n")[0]
        assert py == got, ("perf sanity mismatch", p, small, py, got)

# cross-check the whole suite against the reference binary
by_cmd = {}
bad = 0
for c in cases:
    got = run_rx(c["cmd"], c["input"])[0] if c["ms"] != "r15" else c["expect"]
    if got != c["expect"]:
        bad += 1
        print("MISMATCH", c["id"], repr(c["input"]), "oracle:", c["expect"], "ref:", got)
if bad:
    sys.exit(f"{bad} oracle/ref mismatches")

R = random.Random(15)
public, hidden = [], []
by = {}
for c in cases:
    by.setdefault(c["ms"], []).append(c)
for ms, cs in sorted(by.items()):
    if len(cs) > 20 and ms != "r15":
        keep = set(R.sample(range(len(cs)), 20))
        cs = [c for i, c in enumerate(cs) if i in keep]
    k = 1 if ms == "r15" else max(1, round(len(cs) * 0.1))
    pick = set(R.sample(range(len(cs)), k))
    for i, c in enumerate(cs):
        (public if i in pick else hidden).append(c)
os.makedirs(PUB, exist_ok=True)
json.dump(hidden, open(os.path.join(HID, "cases.json"), "w"))
json.dump(public, open(os.path.join(PUB, "public_cases.json"), "w"), indent=0)
print("hidden", len(hidden), "public", len(public), {m: len(v) for m, v in sorted(by.items())})
