#!/usr/bin/env bash
# Submit one prepared upstream Strata PR: checks the branch (one commit chain on 296b59d2, ASCII, no local paths),
# pushes it to the hotschmoe fork, opens the issue (ISSUE_TITLE.txt / ISSUE_BODY.md) unless ISSUE=<n> is given,
# then the PR (PR_TITLE.txt / PR_BODY.md with "#ISSUE" -> the issue number, "#DEPENDS" -> $DEPENDS).
# Usage: pr_submit.sh <result-dir> <worktree> <branch> <issue-label>      (DRY=1: checks only)
set -euo pipefail
D=${1:?result dir}; W=${2:?worktree}; BR=${3:?branch}; LABEL=${4:?label}
AI="Prepared with AI assistance (Claude Code): the code, the measurements and this text were produced by Claude on the submitter's two Arc Pro B70s, at the submitter's request."
[ "$(git -C "$W" rev-parse --abbrev-ref HEAD)" = "$BR" ] || { echo "worktree not on $BR"; exit 1; }
[ -z "$(git -C "$W" status --short --untracked-files=no)" ] || { echo "worktree dirty"; exit 1; }
git -C "$W" merge-base --is-ancestor 296b59d2 HEAD || { echo "not based on 296b59d2"; exit 1; }
git -C "$W" log --oneline 296b59d2..HEAD | cat
git -C "$W" diff --stat 296b59d2 | tail -1
bad=$(git -C "$W" diff 296b59d2 | grep -E '^\+' | LC_ALL=C grep -c -P '[^\x00-\x7F]|/mnt/|/home/' || true)
[ "$bad" = 0 ] || { echo "diff has $bad non-ASCII / local-path lines"; exit 1; }
for f in PR_TITLE.txt PR_BODY.md; do [ -s "$D/$f" ] || { echo "missing $D/$f"; exit 1; }; done
if grep -n -E 'TODO|/mnt/|/home/' "$D/PR_BODY.md" "$D/PR_TITLE.txt"; then echo "PR text has TODO/local paths"; exit 1; fi
grep -q 'Prepared with AI assistance (Claude Code)' "$D/PR_BODY.md" || { echo "PR body lacks the AI note"; exit 1; }
[ -n "${DRY:-}" ] && { echo "DRY: checks passed"; exit 0; }
git -C "$W" push -q https://github.com/hotschmoe/Strata.git "$BR:$BR" 2>&1 | grep -v '^remote:' || true
if [ -z "${ISSUE:-}" ]; then
  grep -q 'Prepared with AI assistance' "$D/ISSUE_BODY.md" || printf '\n%s\n' "$AI" >> "$D/ISSUE_BODY.md"
  url=$(gh issue create -R Niko1221/Strata --title "$(cat "$D/ISSUE_TITLE.txt")" --body-file "$D/ISSUE_BODY.md" --label "$LABEL" | tail -1)
  ISSUE=${url##*/}; echo "issue: $url"
fi
python3 - "$D" "$ISSUE" "${DEPENDS:-}" <<'EOF'
import sys
from pathlib import Path
D, n, dep = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
s = (D / 'PR_BODY.md').read_text().replace('#ISSUE', f'#{n}')
if dep: s = s.replace('#DEPENDS', f'#{dep}')
assert '#ISSUE' not in s and '#DEPENDS' not in s
(D / 'PR_BODY.final.md').write_text(s.rstrip() + '\n\n\U0001F916 Generated with [Claude Code](https://claude.com/claude-code)\n')
EOF
gh pr create -R Niko1221/Strata --base main --head "hotschmoe:$BR" --title "$(cat "$D/PR_TITLE.txt")" --body-file "$D/PR_BODY.final.md" | tail -1
