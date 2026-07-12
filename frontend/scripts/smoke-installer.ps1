param(
  [Parameter(Mandatory = $true)]
  [string]$InstallerPath,

  [Parameter(Mandatory = $true)]
  [string]$InstallDir
)

$ErrorActionPreference = 'Stop'
$installer = (Resolve-Path -LiteralPath $InstallerPath).Path
$target = [System.IO.Path]::GetFullPath($InstallDir)

if (Test-Path -LiteralPath $target) {
  throw "Smoke-install target already exists: $target"
}

Start-Process -FilePath $installer -ArgumentList @('/S', "/D=$target") -Wait -NoNewWindow

$app = Join-Path $target 'PolyML.exe'
$backendMain = Join-Path $target 'resources\backend\main.py'
$environment = Join-Path $target 'resources\backend\environment.yml'
foreach ($required in @($app, $backendMain, $environment)) {
  if (-not (Test-Path -LiteralPath $required)) {
    throw "Installer smoke test failed; missing required file: $required"
  }
}

Write-Output "Installer smoke test passed: $target"
