# rx specification (version 1): a backtracking regular expression engine

You implement `rx`, a command line program in Rust (standard library only, no external crates) that evaluates
regular expressions written in a precisely defined syntax. The grader compares `rx` with Python's `re` module
(leftmost-first backtracking semantics) on the subset defined here. Where this document is silent inside the
supported subset, Python 3 `re` behaviour on ASCII text is the reference. Anything marked "unspecified" is never tested.

The project is a cargo binary crate named `rx` (`cargo build --release --offline` must produce `target/release/rx`).

## 1. Command line and case file format

```
rx match   CASEFILE      # leftmost match of each case
rx findall CASEFILE      # all non-overlapping matches
rx sub     CASEFILE      # replace all matches using a template
```

`CASEFILE` is a text file made of cases that follow each other directly, each case being a fixed number of lines
(every line, including the last one of the file, ends with a single `\n`):

* `match` and `findall`: 3 lines per case: **pattern**, **flags**, **subject**.
* `sub`: 4 lines per case: **pattern**, **flags**, **template**, **subject**.

Line contents:

* **pattern** and **template** lines are taken literally (no extra escaping; a regular expression backslash is just
  a backslash). A pattern line cannot contain a newline character; use `\n` inside the pattern when needed.
* **flags**: any subset of the letters `i m s x` (see section 4), possibly empty.
* **subject**: the text to search. Written with these escapes (the only place where they apply): `\\` is a backslash,
  `\n` newline, `\t` tab, `\r` carriage return, `\xHH` the byte with two hex digits `HH` (`00` to `7f`). Every other
  character stands for itself. Example: the subject line `a\tb\\n` is the 5-byte text `a`, TAB, `b`, `\`, `n`.
* All text is 7-bit ASCII: pattern, subject and template bytes are `0x00`..`0x7f` (bytes above `0x7f` never occur).
  Positions are byte offsets into the decoded subject, starting at 0.

`rx` prints exactly one output line per case, in order, then exits with status 0. A case never makes the program
fail: invalid patterns produce an `E` line. (Exit status 2 and a message on stderr for a missing file or wrong
command line is fine.)

### Output lines

* `match`: `N` when there is no match. Otherwise `M` followed by one span per group: the whole match first, then
  group 1, 2, ... in order of their opening parenthesis. A span is `START,END` (END exclusive), or `-` for a group
  that did not take part in the match. Example: `M 3,7 3,5 -`.
* `findall`: `F COUNT` followed by the span `START,END` of every match (whole matches only, no groups), separated by
  single spaces; `F 0` when there is no match.
* `sub`: `S ` followed by the resulting text, encoded with the same escapes as the subject (but when writing:
  `\` as `\\`, newline as `\n`, TAB as `\t`, carriage return as `\r`, any other byte below `0x20` or equal to
  `0x7f` as `\xHH` with lower-case hex digits; everything else, including space, as itself). An empty result is `S ` (with the space).
* `E KIND` for an invalid pattern (or template for `sub`), where KIND is one of the error kinds of section 5.

## 2. Pattern syntax

Outermost structure: `alternation := concatenation { "|" concatenation }`, `concatenation := { quantified-atom }`.
An empty alternative matches the empty string (`a|`, `|a`, `()` are valid).

### Atoms

| syntax | matches |
|---|---|
| any other character | itself |
| `.` | any byte except `\n` (any byte at all with flag `s`) |
| `^` `$` | anchors (section 3) |
| `\A` `\Z` `\b` `\B` | anchors / word boundary assertions (section 3) |
| `\d` `\D` | digit `[0-9]` / non-digit |
| `\w` `\W` | word byte `[A-Za-z0-9_]` / non-word |
| `\s` `\S` | white space `[ \t\n\r\f\v]` (space, TAB, LF, CR, FF, VT) / non-white-space |
| `\n \t \r \f \v` | newline, tab, carriage return, form feed (0x0c), vertical tab (0x0b) |
| `\xHH` | the byte `HH` (exactly two hex digits, value `00`..`7f`; fewer or non-hex digits: error `escape`) |
| `\` followed by a punctuation or space character (anything that is not a letter or digit) | that character literally (`\.`, `\*`, `\\`, `\ `, `\-` ...) |
| `\` followed by a letter or digit not listed above or in `\1`..`\9` | error `escape`; a pattern ending in a single `\` is error `escape` |
| `\1` ... `\9` | back reference to group N (section 3) |
| `[...]` | character class |
| `(...)` | capturing group (number = position of its opening parenthesis among all capturing groups, starting at 1) |
| `(?:...)` | non-capturing group |
| `(?P<name>...)` | capturing group that also has a name (`[A-Za-z_][A-Za-z0-9_]*`; a repeated name is error `name`) |
| `(?P=name)` | back reference to the named group (unknown name: `name`) |
| `(?=...)` `(?!...)` | positive / negative lookahead |
| `(?<=...)` `(?<!...)` | positive / negative lookbehind (fixed width, see below) |
| `(?>...)` | atomic group |
| `(?flags)` | global flags, only at the very start of the pattern (section 4) |
| `(?flags:...)` `(?flags-flags:...)` `(?-flags:...)` | scoped flags (section 4) |

A `{` that does not start a valid counted quantifier (section below) is an ordinary character. A `}` is always an
ordinary character. `)` without an open group is error `paren`; a missing `)` is error `paren`.

### Character classes

`[` starts a class, `]` ends it. `^` right after `[` negates the class. A `]` that comes first (after an optional `^`)
is an ordinary member. Members: single characters, ranges `a-z` (both ends single characters, low <= high; otherwise
error `class`), and the escapes `\d \D \w \W \s \S` (which cannot be range ends: `[\d-z]` and `[a-\d]` are error
`class`), `\n \t \r \f \v \xHH`, `\b` (backspace, byte 0x08, only inside classes) and `\` plus a punctuation or space
character (literal). Other letter/digit escapes are error `escape`. A `-` that is first, last, or right after a
range is an ordinary member. A missing `]` is error `class`. A negated class matches every byte that is not a member,
including `\n`. With flag `i`, ASCII letters match both cases.

### Quantifiers

`*` (0 or more), `+` (1 or more), `?` (0 or 1), `{m}`, `{m,}`, `{m,n}` and `{,n}` (= `{0,n}`) where `m` and `n` are
decimal numbers (every other brace form, e.g. `{x}`, `{1`, `{ 1}`, is literal text; `{,}` is unspecified). A quantifier applies to the atom before it
(a group, class, escape, single character, `.`, or lookaround). Suffixes: `?` after a quantifier makes it lazy,
`+` makes it possessive (atomic: after the repeated part has matched, it never gives characters back).
Counted quantifiers must have `m <= n` (else error `repeat`); counts up to 1000 must work.

Error `repeat` also covers: a quantifier with nothing to repeat (at the start of the pattern or an alternative, or
after `(`), a second quantifier directly after a quantified atom (`a**`, `a+*`, `a{2}{3}`; but `a*?` and `a*+` are the
lazy/possessive forms), and quantifying `^ $ \A \Z \b \B` directly (`^*`). Quantifying a group that contains only
an assertion (`(?:^)*`) is allowed.

## 3. Matching semantics

A match is searched at start positions 0, 1, ..., len(subject) in this order; the first start position at which
the pattern can match wins. At that position the match found is the first one in **priority order** of a
backtracking matcher:

* alternation tries the left alternative first;
* greedy quantifiers try the largest repeat count first, lazy ones the smallest; a counted quantifier `x{m,n}`
  matches `x` between m and n times;
* possessive quantifiers and atomic groups `(?>x)` take the first match of `x` in priority order and never
  backtrack into it;
* lookahead `(?=x)` / `(?!x)` test whether `x` matches at the current position (without consuming); the body
  is atomic. A positive lookaround keeps the group captures set inside it; a negative one keeps none.
* lookbehind `(?<=x)` / `(?<!x)` test whether `x` matches the text that ends at the current position. `x` must
  have a fixed width: every alternative the same width, quantifiers only as `{n}` (exact count), no back references;
  otherwise error `lookbehind`. (Groups, classes, anchors and lookarounds inside are fine.)

Captures: a group records the span of its most recent iteration. Captures made in earlier iterations of an
enclosing loop stay set when a later iteration does not set them (`(?:(a)|b)*` on `ab` leaves group 1 = `0,1`).
When the matcher backtracks, captures are restored to their previous values. A group that never took part in the
final match is unset (`-`).

An iteration of `*`, `+`, `{m,}` (beyond the required minimum copies) that matches the empty string ends the loop
(the empty iteration is kept, no further iterations are tried).

Anchors (positions): `^` matches at position 0, and with flag `m` also right after every `\n`. `$` matches at the
end of the subject or just before a final `\n` that is the last byte (so `abc$` matches in `abc\n`), and with flag `m`
also before every `\n`. `\A` matches only at 0; `\Z` only at the very end of the subject. `\b` matches at a position
where exactly one of the previous byte and the next byte is a word byte (`[A-Za-z0-9_]`; the ends of the subject
count as non-word, so `\b` never matches in an empty subject); `\B` matches where `\b` does not.

Back reference `\N` / `(?P=name)` must refer to a group whose closing parenthesis comes earlier in the pattern
(else error `backref`; forward references and references to the group it is inside of are errors; unknown name:
`name`). It matches the text last captured by that group; if the group is unset the back reference fails.
Compared case-insensitively when flag `i` is in effect at the back reference. Only single-digit references `\1`..`\9`
are specified.

## 4. Flags

`i`: ASCII letters match both cases (literals, classes, back references). `m`: multiline anchors. `s`: `.` matches
`\n`. `x`: verbose: white space (space, TAB, LF, CR, FF, VT) in the pattern outside character classes is ignored
and `#` starts a comment running to the end of the pattern line (use `\ ` or `[ ]` for a literal space, `\#` for `#`);
white space between an atom and its quantifier is ignored too.

Flags come from the flags line of the case, from `(?flags)` at the very start of the pattern (letters `imsx`, at
least one, no `-`; anywhere else it is error `flags`), and from scoped groups `(?on-off:...)` with letters
`imsx`: `(?i:...)` turns i on for the group's body, `(?-i:...)` turns it off, `(?i-s:...)` both. Scoped flags
affect the pattern text inside the group, including nested groups, and end at its `)`. An unknown or malformed
flag group is error `flags` (a `(?` followed by anything that is not a defined construct: `group`).

## 5. Errors

Printed as `E KIND`. The kinds, with the situations that produce them:

| KIND | situation |
|---|---|
| `paren` | missing `)`, unmatched `)` |
| `class` | missing `]`, bad range (`z-a`, `\d-z`) |
| `repeat` | nothing to repeat, multiple repeat, `{m,n}` with `m > n`, quantified `^ $ \A \Z \b \B` |
| `escape` | unknown letter/digit escape, trailing `\`, bad or incomplete `\xHH` (a value above 7f is unspecified) |
| `backref` | `\N` to a group that does not exist yet or is still open (the group has to be closed before the back reference) |
| `lookbehind` | lookbehind body not of fixed width |
| `name` | malformed or repeated group name, `(?P=name)` with unknown name |
| `flags` | `(?flags)` not at the start of the pattern, malformed flag groups |
| `group` | `(?` followed by something that is not a defined construct (`(?z)`, `(?<x>...)`, `(?`) |
| `template` | (`sub` only) bad template, see below |

When several problems exist, the one reported is the first met while reading the pattern from left to right
(group-level problems such as a missing `)` are discovered when the end of the pattern is reached). Patterns are
parsed completely before any matching, so an invalid pattern gives `E` even if the subject is empty.

## 6. findall and sub

Matches are found by the following loop (this is Python's `finditer`): `pos = 0`; repeat: search for a match
starting the scan at `pos` (start positions `pos..=len`); if none, stop; record it; set `pos` to its end; if the
match was empty, the next search must not return an empty match at the position `pos` (if the pattern can match
empty there, the matcher must instead backtrack into a non-empty alternative at that same start position, else
move on to start position `pos+1`). A match adjacent to a previous non-empty match may be empty (`a*` on `baac`
gives `0,0 1,3 3,3 4,4`).

`sub` replaces every match of that loop by the expansion of the **template** and copies the text between matches
unchanged. Template syntax: `\1`..`\9` (group N, empty text if unset), `\g<name>` (named group), `\g<N>` (group N,
`\g<0>` is the whole match), `\\` (backslash), `\n` `\t` `\r` (newline, tab, carriage return); every other backslash
sequence, a trailing backslash, a group number larger than the group count or an unknown name is error `template`
(reported even when there is no match; pattern errors take precedence). Everything else is copied literally.
Digits following `\N` are literal text (`\10` is group 1 then `0` -- cases avoid this).

## 7. Performance requirement

Each case must finish within 10 seconds of wall time. The last milestone uses patterns that make a naive
backtracking matcher take exponential time (`(a+)+$`, `(a|aa)+$` ...) on subjects of 40 to 200 characters. Memoising
failed (instruction, position) states of the backtracking search is a standard fix when the pattern has no back
references, lookarounds or atomic groups (RE2's BitState); other approaches (Pike VM, DFA for the plain regular
subset) are also fine. Matching results must still follow the priority order of section 3.

## 8. Examples

```
pattern          flags  subject        match output
a(b+)c           ""     xabbbc         M 1,6 2,5
(a)|(b)          ""     b              M 0,1 - 0,1
^b               m      a\nb           M 2,3
(?i)ab           ""     xAB            M 1,3
(a                ""     a              E paren
```

## 9. Not required

Unicode (non-ASCII) text, locale, `(?#...)` comments, conditional groups, `(?P<name>` with names longer than
64 bytes, `\N` for N >= 10, octal/`\0` escapes, `\u`, `\N{...}`, named group references in lookbehind, recursion, callouts,
possessive/lazy interaction on `{,}` forms, and quantified lookarounds.
