#!/usr/bin/env bash
# Stop hook: if source files under app/ changed in the working tree but nothing
# under tests/ did, block the stop and remind Claude to update the tests.
#
# Loop guard: the reminder fires at most once per unique set of changed src
# files. The set is hashed into a marker under .git; if the same src files are
# still the only thing changed on the next Stop, we don't nag again — so Claude
# can consciously decide "no test needed" and finish.

cd "${CLAUDE_PROJECT_DIR:-.}" 2>/dev/null || exit 0

# Not a git repo -> nothing to compare against.
git rev-parse --git-dir >/dev/null 2>&1 || exit 0

# Changed paths (modified/added/untracked), relative to repo root. $NF handles
# rename lines ("R old -> new") by taking the new path.
paths=$(git status --porcelain 2>/dev/null | awk '{print $NF}')

src_changed=$(printf '%s\n' "$paths" | grep -E '^app/' || true)
tests_changed=$(printf '%s\n' "$paths" | grep -E '^tests/' || true)

# Only remind when src changed and tests did not.
if [ -z "$src_changed" ] || [ -n "$tests_changed" ]; then
  exit 0
fi

git_dir=$(git rev-parse --git-dir 2>/dev/null)
marker="$git_dir/.claude-test-reminder"
state=$(printf '%s' "$src_changed" | git hash-object --stdin 2>/dev/null)

# Already reminded for this exact set of changed src files -> let Claude stop.
if [ -f "$marker" ] && [ "$(cat "$marker" 2>/dev/null)" = "$state" ]; then
  exit 0
fi
printf '%s' "$state" >"$marker"

files=$(printf '%s\n' "$src_changed" | sed 's/^/  - /')
reason="Source files under app/ changed but no tests/ were updated:
$files

Before finishing, update or add test cases in tests/ that cover this change, then run the suite (.venv/bin/python -m pytest -q). If a test genuinely isn't warranted, say so explicitly and you can finish."

# Stop hooks signal "keep going" via decision=block; reason is fed back to Claude.
jq -n --arg reason "$reason" '{decision: "block", reason: $reason}'
exit 0
