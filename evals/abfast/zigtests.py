"""Extract top-level Zig `test "name" {...}` blocks from source text.

Used to build hidden verifiers for repos whose tests live inline in src files:
the blocks that exist in the real commit but not in the parent are appended to
the agent's version of the file before the hidden test run.
"""
import re

_TEST_RE = re.compile(r'^test\s+(?:"((?:[^"\\]|\\.)*)"|([A-Za-z_][A-Za-z0-9_]*))?\s*\{', re.M)


def _match_brace(src, i):
    """src[i] == '{'; return index just past the matching '}'."""
    depth = 0
    n = len(src)
    while i < n:
        c = src[i]
        if c == '/' and src.startswith('//', i):
            j = src.find('\n', i)
            i = n if j < 0 else j
            continue
        if c == '\\' and src.startswith('\\\\', i):  # multiline string line
            j = src.find('\n', i)
            i = n if j < 0 else j
            continue
        if c == '"':
            i += 1
            while i < n and src[i] != '"':
                i += 2 if src[i] == '\\' else 1
        elif c == "'":
            i += 1
            while i < n and src[i] != "'":
                i += 2 if src[i] == '\\' else 1
        elif c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    raise ValueError("unbalanced braces")


def test_blocks(src):
    """Return {name: text} for top-level test blocks, with leading comment lines."""
    out = {}
    for m in _TEST_RE.finditer(src):
        name = m.group(1) or m.group(2) or ""
        end = _match_brace(src, m.end() - 1)
        start = m.start()
        # include directly preceding comment lines
        lines_before = src[:start].split('\n')
        k = len(lines_before) - 1  # last element is '' (start of line)
        pre = []
        while k > 0 and lines_before[k - 1].lstrip().startswith('//'):
            pre.append(lines_before[k - 1])
            k -= 1
        text = ('\n'.join(reversed(pre)) + '\n' if pre else '') + src[start:end]
        out[name] = text
    return out


def new_blocks(parent_src, real_src):
    """Test blocks present (by name) in real_src but not in parent_src, in order."""
    old = test_blocks(parent_src)
    return [t for n, t in test_blocks(real_src).items() if n not in old]


def top_decl(src, name):
    """Text of the top-level `[pub] const|fn|var NAME ...` declaration (with preceding // comments)."""
    m = re.search(r'^(?:pub\s+)?(?:const|var|fn|inline fn)\s+' + re.escape(name) + r'\b', src, re.M)
    if not m:
        raise ValueError("decl not found: " + name)
    first_end = src.find('\n', m.start())
    first = src[m.start():first_end if first_end >= 0 else len(src)]
    if first.rstrip().endswith(';') and '{' not in first:
        end = first_end
    else:
        # the body brace: when the first line ends with '{' it is the last '{' of that line
        # (the return type may itself contain `struct {`/`error{` braces); else the first '{' after the decl
        body = (m.start() + first.rindex('{')) if first.rstrip().endswith('{') else src.index('{', m.start())
        end = _match_brace(src, body)
        while end < len(src) and src[end] in ' \t':
            end += 1
        if end < len(src) and src[end] == ';':
            end += 1
    lines_before = src[:m.start()].split('\n')
    k = len(lines_before) - 1
    pre = []
    while k > 0 and lines_before[k - 1].lstrip().startswith('//'):
        pre.append(lines_before[k - 1]); k -= 1
    return ('\n'.join(reversed(pre)) + '\n' if pre else '') + src[m.start():end]
