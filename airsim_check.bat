@echo off
CALL C:\ProgramData\miniconda3\Scripts\activate.bat C:\Users\Administrator\.conda\envs\client

:loop
REM 运行Python脚本更新端口信息
python airsim_check.py

for /f %%a in (todo_ports.txt) do (
    REM 启动对应端口的Docker容器
    echo Starting Docker container for port %%a...
    docker start "%%a"
    REM 启动新的服务实例
    echo Starting server on port %%a...
    wt new-tab --title "Run AirSim with %%a" cmd /c "C:\Users\Administrator\Documents\UAV\Single_Agent_minigrid\run_airsim.bat C:\Users\Administrator\Documents\UAV\Single_Agent_minigrid\airsim_configs\%%a.json"
    timeout /t 5 /nobreak
)

for /f %%b in (died_ports.txt) do (
    REM 停止 对应端口的Docker容器
    echo Stopping and removing Docker container for port %%b...
    docker stop "%%b"
)

:check
REM 清空todo_ports.txt和died_ports.txt
echo. > todo_ports.txt
echo. > died_ports.txt

REM 提供退出选项
echo Press 'N' to stop or any other key to continue...
choice /C YN /N /D Y /T 30
if errorlevel 2 goto endloop

goto loop

:endloop
echo Stopped by user.
