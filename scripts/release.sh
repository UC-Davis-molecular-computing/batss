#!/usr/bin/env bash
#
# release.sh — create a GitHub Release for the version in Cargo.toml, which
# triggers the workflow to build all platforms AND publish to PyPI.
#
# This DOES publish (irreversibly for that version number). It refuses to run
# unless:
#   * the version in Cargo.toml is strictly newer than the latest release/tag
#     already on GitHub, and
#   * the tag vX.Y.Z does not already exist on the remote, and
#   * your local main matches origin/main (so the release matches what CI tested).
# Then it asks you to type the tag to confirm before creating the release.
#
# Usage:  scripts/release.sh
# Needs:  gh CLI, authenticated (`gh auth login`) with repo + workflow scope.

set -euo pipefail
# If the script dies on an unhandled command failure, say exactly where.
trap 'rc=$?; echo "✗ script failed (exit $rc) near line $LINENO: $BASH_COMMAND" >&2' ERR

echo "==> Checking gh CLI, authentication, and that we're at the repo root..."
command -v gh >/dev/null || { echo "error: gh CLI not found — https://cli.github.com/" >&2; exit 1; }
gh auth status >/dev/null 2>&1 || { echo "error: gh not authenticated — run: gh auth login" >&2; exit 1; }
[ -f Cargo.toml ] || { echo "error: run from the repo root (Cargo.toml not found)" >&2; exit 1; }

echo "==> Reading the package version from Cargo.toml..."
VERSION=$(grep -m1 -E '^version[[:space:]]*=[[:space:]]*"' Cargo.toml | sed -E 's/.*"([^"]+)".*/\1/')
[ -n "$VERSION" ] || { echo "error: couldn't read package version from Cargo.toml" >&2; exit 1; }
TAG="v$VERSION"
echo "    Cargo.toml version: $VERSION   (tag: $TAG)"

# strict-greater semver test: ver_gt A B  ==> true if A > B
ver_gt() { [ "$1" != "$2" ] && [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | tail -1)" = "$1" ]; }

echo "==> Fetching origin/main (branch only, not --tags: a local tag that differs"
echo "    from the remote's would make 'git fetch --tags' fail and abort this script)..."
git fetch origin --quiet || { echo "error: 'git fetch origin' failed — cannot verify you are in sync with origin/main." >&2; exit 1; }

echo "==> Verifying local main == origin/main (so CI builds exactly what you tested)..."
if [ "$(git rev-parse HEAD)" != "$(git rev-parse origin/main)" ]; then
  echo "error: local HEAD != origin/main. Commit and push your changes first." >&2
  exit 1
fi
git diff --quiet HEAD -- || echo "    warning: working tree has uncommitted changes; they will NOT be in the release."

echo "==> Finding the latest existing release/tag on GitHub..."
LATEST=$(gh release view --json tagName --jq .tagName 2>/dev/null || true)
[ -n "$LATEST" ] || LATEST=$(git tag -l 'v*' | sort -V | tail -1)
echo "    Latest release/tag on GitHub: ${LATEST:-<none>}"

echo "==> Guard: tag $TAG must not already exist on origin..."
if git ls-remote --tags origin "refs/tags/$TAG" | grep -q "refs/tags/$TAG"; then
  echo "error: tag $TAG already exists on origin. Bump the version in Cargo.toml" >&2
  echo "       (and delete the tag + release first if this is a re-do)." >&2
  exit 1
fi
echo "==> Guard: $VERSION must be strictly newer than the latest release..."
if [ -n "$LATEST" ] && ! ver_gt "$VERSION" "${LATEST#v}"; then
  echo "error: $VERSION is NOT newer than the latest release ${LATEST#v}." >&2
  echo "       Bump the version in Cargo.toml (semver) before releasing." >&2
  exit 1
fi

echo
echo "This will create GitHub Release $TAG on origin/main and PUBLISH $VERSION to PyPI."
echo "A PyPI version cannot be re-uploaded once published."
read -rp "Type the tag ($TAG) to proceed, anything else to abort: " CONFIRM
[ "$CONFIRM" = "$TAG" ] || { echo "Aborted."; exit 1; }

echo "==> Recording the latest release-triggered run (so we can spot the new one)..."
before=$(gh run list --workflow=build_and_publish.yml --event release \
         --limit 1 --json databaseId --jq '.[0].databaseId // ""' 2>/dev/null || true)

echo "==> Creating GitHub Release $TAG (this triggers the build + publish workflow)..."
gh release create "$TAG" --target main --title "$TAG" --generate-notes
echo "    Created release $TAG."

echo "==> Waiting for the build+publish run to register..."
RID=""
for _ in $(seq 1 40); do
  cur=$(gh run list --workflow=build_and_publish.yml --event release \
        --limit 1 --json databaseId --jq '.[0].databaseId // ""' 2>/dev/null || true)
  if [ -n "$cur" ] && [ "$cur" != "$before" ]; then RID="$cur"; break; fi
  sleep 3
done
if [ -z "$RID" ]; then
  echo "Release created, but couldn't locate the workflow run — check the Actions tab." >&2
  exit 0
fi

echo "==> Run: $(gh run view "$RID" --json url --jq .url)"
echo "==> Watching build + publish to completion..."
if gh run watch "$RID" --exit-status --compact; then
  echo "✓ Success. Verify at: https://pypi.org/project/batss/$VERSION/"
else
  echo "✗ FAILED. To retry after fixing, delete BOTH the release AND the tag $TAG:" >&2
  echo "    gh release delete $TAG --yes --cleanup-tag" >&2
  exit 1
fi
