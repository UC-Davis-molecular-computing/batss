#!/usr/bin/env pwsh
#
# release.ps1 - create a GitHub Release for the version in Cargo.toml, which
# triggers the workflow to build all platforms AND publish to PyPI.
# PowerShell mirror of scripts/release.sh.
#
# This DOES publish (irreversibly for that version). It refuses to run unless:
#   * the Cargo.toml version is strictly newer than the latest release/tag on GitHub,
#   * the tag vX.Y.Z does not already exist on the remote, and
#   * local main matches origin/main (so CI builds exactly what you tested).
# Then it asks you to type the tag to confirm.
#
# Usage:  ./scripts/release.ps1
# Needs:  gh CLI (authenticated, repo + workflow scope) and git.

Write-Host '==> Checking gh CLI, authentication, and repo root...'
if (-not (Get-Command gh -ErrorAction SilentlyContinue)) { Write-Error 'gh CLI not found - https://cli.github.com/'; exit 1 }
gh auth status *> $null
if ($LASTEXITCODE -ne 0) { Write-Error 'gh not authenticated - run: gh auth login'; exit 1 }
if (-not (Test-Path Cargo.toml)) { Write-Error 'run from the repo root (Cargo.toml not found)'; exit 1 }

Write-Host '==> Reading the package version from Cargo.toml...'
$m = Select-String -Path Cargo.toml -Pattern '^version\s*=\s*"([^"]+)"' | Select-Object -First 1
if (-not $m) { Write-Error "couldn't read package version from Cargo.toml"; exit 1 }
$Version = $m.Matches[0].Groups[1].Value
$Tag = "v$Version"
Write-Host "    Cargo.toml version: $Version   (tag: $Tag)"

# strict-greater semver test: returns $true if A > B
function Test-VersionGreater([string]$a, [string]$b) {
    if ($a -eq $b) { return $false }
    $pa = @($a.Split('.') | ForEach-Object { [int]$_ })
    $pb = @($b.Split('.') | ForEach-Object { [int]$_ })
    for ($i = 0; $i -lt [Math]::Max($pa.Count, $pb.Count); $i++) {
        $x = if ($i -lt $pa.Count) { $pa[$i] } else { 0 }
        $y = if ($i -lt $pb.Count) { $pb[$i] } else { 0 }
        if ($x -gt $y) { return $true }
        if ($x -lt $y) { return $false }
    }
    return $false
}

Write-Host '==> Fetching origin/main (branch only, not --tags: a local tag that differs from the remote would fail the fetch)...'
git fetch origin --quiet
if ($LASTEXITCODE -ne 0) { Write-Error "'git fetch origin' failed - cannot verify sync with origin/main."; exit 1 }

Write-Host '==> Verifying local main == origin/main...'
if ((git rev-parse HEAD).Trim() -ne (git rev-parse origin/main).Trim()) {
    Write-Error 'local HEAD != origin/main. Commit and push first so CI builds what you tested.'; exit 1
}
git diff --quiet HEAD --
if ($LASTEXITCODE -ne 0) { Write-Warning 'working tree has uncommitted changes; they will NOT be in the release.' }

Write-Host '==> Finding the latest existing release/tag on GitHub...'
$Latest = (gh release view --json tagName --jq .tagName 2>$null | Out-String).Trim()
if (-not $Latest) { $Latest = git tag -l 'v*' | Sort-Object { [version]($_ -replace '^v', '') } | Select-Object -Last 1 }
Write-Host "    Latest release/tag on GitHub: $(if ($Latest) { $Latest } else { '<none>' })"

Write-Host "==> Guard: tag $Tag must not already exist on origin..."
if (git ls-remote --tags origin "refs/tags/$Tag") {
    Write-Error "tag $Tag already exists on origin. Bump the version in Cargo.toml (delete tag+release first if re-doing)."; exit 1
}
Write-Host "==> Guard: $Version must be strictly newer than the latest release..."
if ($Latest -and -not (Test-VersionGreater $Version ($Latest -replace '^v', ''))) {
    Write-Error "$Version is NOT newer than the latest release $($Latest -replace '^v',''). Bump the version in Cargo.toml."; exit 1
}

Write-Host ''
Write-Host "This will create GitHub Release $Tag on origin/main and PUBLISH $Version to PyPI (cannot be re-uploaded)."
$confirm = Read-Host "Type the tag ($Tag) to proceed, anything else to abort"
if ($confirm -ne $Tag) { Write-Host 'Aborted.'; exit 1 }

Write-Host '==> Recording the latest release-triggered run (so we can spot the new one)...'
$before = (gh run list --workflow=build_and_publish.yml --event release --limit 1 --json databaseId --jq '.[0].databaseId // ""' 2>$null | Out-String).Trim()

Write-Host "==> Creating GitHub Release $Tag (triggers the build + publish workflow)..."
gh release create $Tag --target main --title $Tag --generate-notes
if ($LASTEXITCODE -ne 0) { Write-Error 'gh release create failed'; exit 1 }
Write-Host "    Created release $Tag."

Write-Host '==> Waiting for the build+publish run to register...'
$rid = $null
for ($i = 0; $i -lt 40; $i++) {
    $cur = (gh run list --workflow=build_and_publish.yml --event release --limit 1 --json databaseId --jq '.[0].databaseId // ""' 2>$null | Out-String).Trim()
    if ($cur -and $cur -ne $before) { $rid = $cur; break }
    Start-Sleep -Seconds 3
}
if (-not $rid) { Write-Warning 'release created but run not found - check the Actions tab.'; exit 0 }

Write-Host "==> Run: $(gh run view $rid --json url --jq .url)"
Write-Host '==> Watching build + publish to completion...'
gh run watch $rid --exit-status --compact
if ($LASTEXITCODE -eq 0) {
    Write-Host "Success. Verify at: https://pypi.org/project/batss/$Version/"
}
else {
    Write-Error "FAILED. To retry after fixing, delete BOTH the release and tag: gh release delete $Tag --yes --cleanup-tag"; exit 1
}
