# One-time setup: puts a "Market Forecaster" icon on your desktop and in the Start menu.
# Keep this file in the forecaster folder, next to "Start Market Forecaster.bat".
$folder = $PSScriptRoot
$target = Join-Path $folder "Start Market Forecaster.bat"
$icon   = Join-Path $folder "forecaster.ico"

if (-not (Test-Path $target)) { Write-Host "Can't find 'Start Market Forecaster.bat' in $folder"; exit 1 }
if (-not (Test-Path $icon))   { Write-Host "Can't find 'forecaster.ico' in $folder"; exit 1 }

$shell = New-Object -ComObject WScript.Shell
$places = @([Environment]::GetFolderPath("Desktop"), [Environment]::GetFolderPath("Programs"))

foreach ($place in $places) {
    $lnk = $shell.CreateShortcut((Join-Path $place "Market Forecaster.lnk"))
    $lnk.TargetPath       = $target
    $lnk.WorkingDirectory = $folder
    $lnk.IconLocation     = "$icon,0"
    $lnk.WindowStyle      = 7          # start minimised, so only the browser shows
    $lnk.Description      = "Open the Market Forecaster dashboard"
    $lnk.Save()
    Write-Host "Shortcut created in $place"
}
Write-Host "Done. Double-click 'Market Forecaster' on your desktop to start the app."
