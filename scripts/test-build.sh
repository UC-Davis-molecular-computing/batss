#!/usr/bin/env bash
#
# test-build.sh — build every platform WITHOUT publishing to PyPI.
#
# Triggers the build_and_publish workflow via workflow_dispatch and watches it.
# The workflow's publish job is gated with `if: github.event_name == 'release'`,
# so a workflow_dispatch run builds Linux/musllinux/Windows/macOS + sdist and
# uploads NOTHING to PyPI. Use this to confirm all platforms compile — no GitHub
# release, no tag, no cleanup.
#
# Usage:  scripts/test-build.sh [branch]      (branch defaults to "main")
# Needs:  gh CLI, authenticated (`gh auth login`), with workflow scope.

set -euo pipefail

WORKFLOW="build_and_publish.yml"
REF="${1:-main}"

command -v gh >/dev/null || { echo "error: gh CLI not found — https://cli.github.com/" >&2; exit 1; }
gh auth status >/dev/null 2>&1 || { echo "error: gh not authenticated — run: gh auth login" >&2; exit 1; }

# Record the newest existing dispatch run BEFORE triggering. Right after dispatch,
# GitHub briefly still returns the PREVIOUS run from `gh run list`; watching that
# (already-finished) run is what makes the script look like it "succeeds instantly".
# So we wait for a run whose id differs from this pre-dispatch latest.
before=$(gh run list --workflow="$WORKFLOW" --branch "$REF" --event workflow_dispatch \
         --limit 1 --json databaseId --jq '.[0].databaseId // ""' 2>/dev/null || true)

echo "Dispatching $WORKFLOW on '$REF' (build-only; nothing is published)..."
gh workflow run "$WORKFLOW" --ref "$REF"

# Poll until a genuinely NEW run appears (id != the pre-dispatch latest).
printf 'Locating the new run'
RID=""
for _ in $(seq 1 30); do
  cur=$(gh run list --workflow="$WORKFLOW" --branch "$REF" --event workflow_dispatch \
        --limit 1 --json databaseId --jq '.[0].databaseId // ""' 2>/dev/null || true)
  if [ -n "$cur" ] && [ "$cur" != "$before" ]; then RID="$cur"; break; fi
  printf '.'; sleep 2
done
echo
[ -n "$RID" ] || { echo "error: couldn't find the dispatched run; check 'gh run list --workflow=$WORKFLOW'" >&2; exit 1; }

echo "Run: $(gh run view "$RID" --json url --jq .url)"
echo "Watching to completion (Ctrl-C stops watching, not the run)..."
if gh run watch "$RID" --exit-status --compact; then status=0; else status=$?; fi

echo
echo "=== per-job results ==="
gh run view "$RID" --json status,conclusion,jobs \
  --jq '"overall: \(.status)/\(.conclusion)\n" + ([.jobs[] | "  [\(.conclusion // .status)] \(.name)"] | join("\n"))'
exit "$status"
