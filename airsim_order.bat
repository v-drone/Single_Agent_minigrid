@echo off
powershell -Command "Get-Process | Where-Object { $_.MainWindowTitle -like 'AirSimAssets' } | ForEach-Object {
  Add-Type -Namespace Win32 -Name User32 -UsingNamespace System.Runtime.InteropServices -MemberDefinition '
    [DllImport(\"user32.dll\")]
    public static extern bool MoveWindow(IntPtr hWnd, int X, int Y, int nWidth, int nHeight, bool bRepaint);
  '
  $screen = [System.Windows.Forms.Screen]::PrimaryScreen.WorkingArea
  $width = $screen.Width / 2
  $height = $screen.Height / 2
  [Win32.User32]::MoveWindow($_.MainWindowHandle, 0, 0, $width, $height, $true)
}"
pause
