"""Python `re` oracle producing rx output lines (host python 3.12+; recorded at build time only)."""
import re

FL = {"i": re.I, "m": re.M, "s": re.S, "x": re.X}


def enc(s):
    out = []
    for ch in s:
        o = ord(ch)
        if ch == "\\": out.append("\\\\")
        elif ch == "\n": out.append("\\n")
        elif ch == "\t": out.append("\\t")
        elif ch == "\r": out.append("\\r")
        elif 0x20 <= o <= 0x7e: out.append(ch)
        else: out.append("\\x%02x" % o)
    return "".join(out)


def kind(e):
    m = str(e)
    for key, k in (("missing ), unterminated", "paren"), ("unbalanced parenthesis", "paren"), ("unterminated character set", "class"),
                   ("bad character range", "class"), ("nothing to repeat", "repeat"), ("multiple repeat", "repeat"),
                   ("min repeat greater", "repeat"), ("bad escape", "escape"), ("incomplete escape", "escape"),
                   ("invalid group reference", "backref"), ("cannot refer to an open group", "backref"),
                   ("look-behind requires", "lookbehind"), ("bad character in group name", "name"), ("redefinition of group name", "name"),
                   ("missing group name", "name"), ("unknown group name", "name"), ("global flags not at the start", "flags"),
                   ("unknown extension", "group"), ("missing -, : or )", "flags"), ("bad inline flags", "flags"),
                   ("unterminated name", "name"), ("missing >, unterminated name", "name")):
        if key in m:
            return k
    raise RuntimeError("unmapped re.error: " + m)


def compile_(pat, flags):
    f = 0
    for c in flags:
        f |= FL[c]
    return re.compile(pat, f)


def match_line(pat, flags, subj):
    try:
        rx = compile_(pat, flags)
    except re.error as e:
        return "E " + kind(e)
    m = rx.search(subj)
    if not m:
        return "N"
    out = ["M"]
    for g in range(rx.groups + 1):
        s = m.span(g)
        out.append("-" if s[0] < 0 else "%d,%d" % s)
    return " ".join(out)


def findall_line(pat, flags, subj):
    try:
        rx = compile_(pat, flags)
    except re.error as e:
        return "E " + kind(e)
    ms = list(rx.finditer(subj))
    return " ".join(["F %d" % len(ms)] + ["%d,%d" % m.span() for m in ms])


def sub_line(pat, flags, tmpl, subj):
    try:
        rx = compile_(pat, flags)
    except re.error as e:
        return "E " + kind(e)
    try:
        return "S " + enc(rx.sub(tmpl, subj))
    except (re.error, IndexError):
        return "E template"
