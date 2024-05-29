@echo off
CALL C:\ProgramData\miniconda3\Scripts\activate.bat C:\Users\Administrator\.conda\envs\client

:loop
REM 运行Python脚本更新端口信息
python airsim_check.py

REM 读取待启动的端口列表
setlocal enabledelayedexpansion
set "cmdline="
for /f %%a in (todo_ports.txt) do (
    REM 构建启动新服务实例的命令
    set "cmdline=!cmdline! new-tab --title "Run AirSim with %%a" cmd /c "C:\Users\Administrator\Documents\UAV\Single_Agent_minigrid\run_airsim.bat C:\Users\Administrator\Documents\UAV\Single_Agent_minigrid\airsim_configs\%%a.json" & timeout /t 5 &"
)

REM 执行命令
if not "%cmdline%"=="" (
    wt %cmdline%
)

echo. > todo_ports.txt

REM 提供退出选项
echo Press 'N' to stop or any other key to continue...
choice /C YN /N /D Y /T 30
if errorlevel 2 goto endloop

goto loop

:endloop
echo Stopped by user.
