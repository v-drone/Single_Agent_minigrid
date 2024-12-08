Write-Host "Writing to backup_ports.txt"
5020..5100 | ForEach-Object { $_ } | Set-Content "backup_ports.txt"
Clear-Content todo_ports.txt
Clear-Content died_ports.txt

Write-Host "Ports backup completed."

Start-Sleep -Seconds 1

do
{
    python airsim_check.py

    Get-Content todo_ports.txt | ForEach-Object {
        $port = $_
        Write-Host "Starting server on port $port..."

        $scriptBlock = {
            param($port)
            & powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\Users\Administrator\Documents\UAV\Single_Agent_minigrid\run_airsim_car.ps1" $port
        }
        Start-Job -ScriptBlock $scriptBlock -ArgumentList $port

        Start-Sleep -Seconds 30
    }


    Get-Content died_ports.txt | ForEach-Object {
        $port = $_
        Write-Host "Stopping and removing Docker container for port $port..."
        docker stop "$port"
    }

    Clear-Content todo_ports.txt
    Clear-Content died_ports.txt

    Write-Host "Waiting for 20 seconds. Press any key to exit."
    $startTime = Get-Date
    while ((New-TimeSpan -Start $startTime -End (Get-Date)).TotalSeconds -lt 20 -and -not [Console]::KeyAvailable)
    {
        Start-Sleep -Milliseconds 2000
    }
    if ([Console]::KeyAvailable)
    {
        $host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") | Out-Null
        break
    }
} while ($true)

Write-Host "Stopped by user."
