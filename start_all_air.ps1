$ErrorActionPreference = 'SilentlyContinue'

Write-Host "Writing to backup_ports.txt"
5000..5400 | ForEach-Object { $_ } | Set-Content "backup_ports.txt"
Write-Host "Ports backup completed."

Start-Sleep -Seconds 3

do
{
    python airsim_check.py

    Get-Content todo_ports.txt | ForEach-Object {
        $port = $_
        Write-Host "Starting Docker container for port $port..."
        docker start "$port"

        Write-Host "Starting server on port $port..."
        # 使用 PowerShell Jobs 正确地传递参数
        $scriptBlock = {
            param($port)
            $configPath = "C:\Users\Administrator\Documents\UAV\Single_Agent_minigrid\airsim_configs\$port.json"
            $logPath = "C:\Users\Administrator\Documents\UAV\Single_Agent_minigrid\Logs\$port.log"
            & powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\Users\Administrator\Documents\UAV\Single_Agent_minigrid\run_airsim.ps1" $configPath $logPath
        }
        Start-Job -ScriptBlock $scriptBlock -ArgumentList $port

        Start-Sleep -Seconds 5
    }

    Get-Content died_ports.txt | ForEach-Object {
        $port = $_
        Write-Host "Stopping and removing Docker container for port $port..."
        docker stop "$port"
    }

    Clear-Content todo_ports.txt
    Clear-Content died_ports.txt

    Write-Host "Waiting for 10 seconds. Press any key to exit."
    $startTime = Get-Date
    while ((New-TimeSpan -Start $startTime -End (Get-Date)).TotalSeconds -lt 10 -and -not [Console]::KeyAvailable) {
        Start-Sleep -Milliseconds 500
    }
    if ([Console]::KeyAvailable) {
        $host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") | Out-Null
        break
    }
} while ($true)

Write-Host "Stopped by user."
