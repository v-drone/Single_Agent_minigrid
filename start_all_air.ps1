# 禁用回显
$ErrorActionPreference = 'SilentlyContinue'

# 激活 Python 环境
& C:\ProgramData\miniconda3\Scripts\Activate.ps1 C:\Users\Administrator\.conda\envs\client

# 创建并覆盖 backup_ports.txt 文件
Write-Host "Writing to backup_ports.txt"
5000..5400 | ForEach-Object { $_ } | Set-Content "backup_ports.txt"
Write-Host "Ports backup completed."

# 等待3秒
Start-Sleep -Seconds 3

# 主循环
do
{
    # 运行Python脚本更新端口信息
    python airsim_check.py

    # 处理todo_ports.txt文件中的端口
    Get-Content todo_ports.txt | ForEach-Object {
        $port = $_
        Write-Host "Starting Docker container for port $port..."
        docker start "$port"

        Write-Host "Starting server on port $port..."
        Start-Process -FilePath "cmd" -ArgumentList "/c", "C:\Users\Administrator\Documents\UAV\Single_Agent_minigrid\run_airsim.bat C:\Users\Administrator\Documents\UAV\Single_Agent_minigrid\airsim_configs\$port.json" -NoNewWindow

        Start-Sleep -Seconds 5
    }

    # 处理died_ports.txt文件中的端口
    Get-Content died_ports.txt | ForEach-Object {
        $port = $_
        Write-Host "Stopping and removing Docker container for port $port..."
        docker stop "$port"
    }

    # 清空todo_ports.txt和died_ports.txt
    Clear-Content todo_ports.txt
    Clear-Content died_ports.txt

    # 提供退出选项
    Write-Host "Press 'N' to stop or any other key to continue..."
    $choice = $host.ui.RawUI.ReadKey("NoEcho,IncludeKeyDown")
} while ($choice.Character -ne 'N')

Write-Host "Stopped by user."
