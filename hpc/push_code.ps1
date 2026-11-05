# Push ALL code (ImageSegmentation + ShapeAnalysis, scripts and hpc) to UBELIX and prove that it arrived identical.
#   . "<Code>\ImageSegmentation\hpc\env.ps1"      (sets $code, $ub, $Py)
#   .\..\hpc\push_code.ps1  or  & "$code\ImageSegmentation\hpc\push_code.ps1"        (3 password prompts)
# What it does: one tar of the four code folders -> permissions repaired -> unpack -> Windows line endings removed ->
# a fingerprint list is built on the cluster and compared with the one built here.
$ErrorActionPreference = "Stop"
if (-not $code -or -not $ub -or -not $Py) { throw "run env.ps1 first (it sets `$code, `$ub and `$Py)" }
$here = Join-Path $code "ImageSegmentation\hpc"
$tgz = Join-Path $env:TEMP "code_push.tgz"

$localHash = (& $Py (Join-Path $here "code_manifest.py") --root $code --hash).Trim()
$files = (& $Py (Join-Path $here "code_manifest.py") --root $code | Measure-Object -Line).Lines
Write-Host "local code: $files files, fingerprint $localHash"

tar -czf $tgz --exclude=__pycache__ --exclude=_archive -C $code ImageSegmentation/scripts ImageSegmentation/hpc ShapeAnalysis/scripts ShapeAnalysis/hpc
if ($LASTEXITCODE -ne 0) { throw "tar failed" }
Write-Host ("archive: {0:N1} MB" -f ((Get-Item $tgz).Length / 1MB))

ssh $ub "mkdir -p ~/code && chmod -R u+rwX ~/code"
scp $tgz "${ub}:~/code_push.tgz"
if ($LASTEXITCODE -ne 0) { throw "scp failed" }
$out = ssh $ub "tar -xzf ~/code_push.tgz -C ~/code && rm ~/code_push.tgz && bash ~/code/ImageSegmentation/hpc/after_push.sh"
$out | ForEach-Object { Write-Host "  cluster: $_" }
Remove-Item $tgz

$remoteHash = (($out | Where-Object { $_ -like "manifest:*" }) -replace "manifest:\s*", "").Trim()
if ($remoteHash -eq $localHash) {
    Write-Host "OK: the code on UBELIX is identical to the code here ($localHash)" -ForegroundColor Green
} else {
    Write-Host "MISMATCH: local $localHash, cluster $remoteHash" -ForegroundColor Red
    Write-Host "files that differ or exist on only one side:"
    $remote = ssh $ub "cat ~/code/CODE_MANIFEST.md5"
    $mine = & $Py (Join-Path $here "code_manifest.py") --root $code
    Compare-Object ($mine) ($remote) | ForEach-Object { "  {0}  {1}" -f $_.SideIndicator, $_.InputObject }
}
