@echo off
CALL C:\ProgramData\miniconda3\Scripts\activate.bat C:\Users\Administrator\.conda\envs\client

:loop
REM 运行Python脚本更新端口信息
python airsim_check.py



for /f %%port in (todo_ports.txt) do (
    REM 启动新的服务实例
    echo Starting server on port %%port...
    wt new-tab --title "Run AirSim with %%port" cmd /c "C:\Users\Administrator\Documents\UAV\Single_Agent_minigrid\run_airsim.bat C:\Users\Administrator\Documents\UAV\Single_Agent_minigrid\airsim_configs\%%port.json"
    REM 等待5秒后启动下一个实例
    timeout /t 5 /nobreak
)

:check
REM 清空todo_ports.txt
echo. > todo_ports.txt

REM 提供退出选项
echo Press 'N' to stop or any other key to continue...
choice /C YN /N /D Y /T 30
if errorlevel 2 goto endloop

goto loop

:endloop
echo Stopped by user.
