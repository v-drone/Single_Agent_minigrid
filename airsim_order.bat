@echo off
powershell -Command "& {
$windows = Get-Process | Where-Object { $_.MainWindowTitle -like 'AirSimAssets.exe' } | Select-Object MainWindowHandle
foreach ($window in $windows) {
  $handle = $window.MainWindowHandle
  [void][System.Reflection.Assembly]::LoadWithPartialName('System.Windows.Forms')
  $screen = [System.Windows.Forms.Screen]::PrimaryScreen.WorkingArea
  $null = Add-Type -Namespace Win32 -Name Functions -MemberDefinition '
    [DllImport(\"user32.dll\")]
    public static extern bool MoveWindow(IntPtr hWnd, int X, int Y, int nWidth, int nHeight, bool bRepaint);
  '
  $x = 0; $y = 0; $width = $screen.Width / 2; $height = $screen.Height / 2
  [Win32.Functions]::MoveWindow($handle, $x, $y, $width, $height, $true)
}
}"
pause
