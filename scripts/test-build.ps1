#!/usr/bin/env pwsh
#
# test-build.ps1 - build every platform WITHOUT publishing to PyPI.
# PowerShell mirror of scripts/test-build.sh.
#
# Dispatches the build_and_publish workflow via workflow_dispatch and watches it.
# The publish job is gated on `if: github.event_name == 'release'`, so a
# workflow_dispatch run builds Linux/musllinux/Windows/macOS + sdist and uploads
# NOTHING to PyPI - no release, no tag.
#
# Usage:  ./scripts/test-build.ps1 [branch]      (branch defaults to "main")
# Needs:  gh CLI, authenticated (`gh auth login`) with the workflow scope.

$Workflow = 'build_and_publish.yml'
$Ref = if ($args.Count -ge 1) { $args[0] } else { 'main' }

Write-Host '==> Checking gh CLI is installed and authenticated...'
if (-not (Get-Command gh -ErrorAction SilentlyContinue)) { Write-Error 'gh CLI not found - https://cli.github.com/'; exit 1 }
gh auth status *> $null
if ($LASTEXITCODE -ne 0) { Write-Error 'gh not authenticated - run: gh auth login'; exit 1 }

Write-Host '==> Recording the latest existing run (so we can spot the new one)...'
$before = (gh run list --workflow=$Workflow --branch $Ref --event workflow_dispatch --limit 1 --json databaseId --jq '.[0].databaseId // ""' 2>$null | Out-String).Trim()

Write-Host "==> Dispatching $Workflow on '$Ref' (build-only; nothing is published)..."
gh workflow run $Workflow --ref $Ref
if ($LASTEXITCODE -ne 0) { Write-Error 'workflow dispatch failed'; exit 1 }

Write-Host -NoNewline '==> Waiting for the new run to register (GitHub briefly returns the old one)'
$rid = $null
for ($i = 0; $i -lt 30; $i++) {
    $cur = (gh run list --workflow=$Workflow --branch $Ref --event workflow_dispatch --limit 1 --json databaseId --jq '.[0].databaseId // ""' 2>$null | Out-String).Trim()
    if ($cur -and $cur -ne $before) { $rid = $cur; break }
    Write-Host -NoNewline '.'; Start-Sleep -Seconds 2
}
Write-Host ''
if (-not $rid) { Write-Error "couldn't find the dispatched run; try: gh run list --workflow=$Workflow"; exit 1 }

Write-Host "==> Run: $(gh run view $rid --json url --jq .url)"
Write-Host '==> Watching to completion (Ctrl-C stops watching, not the run)...'
gh run watch $rid --exit-status --compact
$status = $LASTEXITCODE

Write-Host "`n==> Per-job results:"
gh run view $rid --json status,conclusion,jobs --jq '"overall: \(.status)/\(.conclusion // "-")\n" + ([.jobs[] | "  [\(.conclusion // .status)] \(.name)"] | join("\n"))'
exit $status
