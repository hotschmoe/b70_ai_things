# Run a local .sh script on the b70 Unraid box cleanly, with optional env vars.
# Strips BOM/CRLF, prepends `export KEY=VALUE` for any KEY=VALUE args, base64-encodes
# (avoids Windows<->ssh encoding issues), decodes + executes on the remote under bash.
# Usage: powershell -File runremote.ps1 scripts\foo.sh [KEY=VALUE ...] [host=<sshhost>]
param(
  [Parameter(Mandatory = $true)][string]$ScriptPath,
  [Parameter(ValueFromRemainingArguments = $true)][string[]]$Rest
)
$ErrorActionPreference = "Stop"
$SshHost = "b70"
$exports = ""
foreach ($a in $Rest) {
  if ($a -match '^host=(.+)$') { $SshHost = $Matches[1]; continue }
  if ($a -match '^([A-Za-z_][A-Za-z0-9_]*)=(.*)$') {
    $exports += "export $($Matches[1])=" + "'" + ($Matches[2] -replace "'", "'\''") + "'`n"
  }
}
$raw = [System.IO.File]::ReadAllText((Resolve-Path $ScriptPath))
$raw = $raw -replace "`r`n", "`n"
$raw = $raw.TrimStart([char]0xFEFF)
# Insert exports after a shebang line if present, else at top.
if ($raw -match '^#!.*\n') {
  $raw = $raw -replace '^(#![^\n]*\n)', "`$1$exports"
} else {
  $raw = $exports + $raw
}
$bytes = [System.Text.Encoding]::UTF8.GetBytes($raw)
$b64 = [System.Convert]::ToBase64String($bytes)
if ($b64.Length -gt 100000) { throw "Script too large for inline transport ($($b64.Length) b64 chars)" }
ssh $SshHost "echo $b64 | base64 -d | bash -s"
