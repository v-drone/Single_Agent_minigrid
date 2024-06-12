# 禁用回显
$ErrorActionPreference = 'SilentlyContinue'

# 激活 Python 环境
& C:\ProgramData\miniconda3\Scripts\Activate.ps1 C:\Users\Administrator\.conda\envs\client

# 切换目录
Set-Location -Path "C:\Users\Administrator\Documents\UAV\Single_Agent_minigrid"

# 执行 Python 脚本，传递配置文件参数
& python "airsim_client\airsim_runner.py" -f $args[0]

# 输出完成任务的消息
Write-Host "`nTask completed with config: $($args[0])"

# 暂停，等待用户输入，模拟 pause 命令
Write-Host "Press any key to close this window..."
$host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") | Out-Null
