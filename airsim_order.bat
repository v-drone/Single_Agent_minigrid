@echo off
powershell -Command "$procs = Get-Process; foreach ($proc in $procs) { if ($proc.MainWindowTitle -like '*AirSimAssets*') { Add-Type -TypeDefinition '[DllImport(\"user32.dll\")] public static extern bool MoveWindow(IntPtr hWnd, int X, int Y, int Width, int Height, bool Repaint);'; $screen = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds; $w = $screen.Width / 2; $h = $screen.Height / 2; [Win32.User32]::MoveWindow($proc.MainWindowHandle, 0, 0, $w, $h, $true) } }"
pause
