@echo off
CALL C:\ProgramData\miniconda3\Scripts\activate.bat C:\Users\Administrator\.conda\envs\client

REM 遍历airsim_configs目录下特定的配置文件
for %%i in (5000 5001 5002 5003 5004 5005 5006 5007 5008) do (
    REM 在新的Windows Terminal标签中启动run_airsim.bat，传递配置文件路径
    start wt new-tab --title "Run AirSim with %%i" cmd /c "C:\Users\Administrator\Documents\UAV\Single_Agent_minigrid\run_airsim.bat C:\Users\Administrator\Documents\UAV\Single_Agent_minigrid\airsim_configs\%%i.json"
)

echo All specified AirSim instances have been started.

REM 创建并覆盖 backup_ports.txt 文件
echo Writing to backup_ports.txt
(for /l %%j in (5009,1,5020) do echo %%j) > backup_ports.txt

echo Ports backup completed.

REM 等待10秒
timeout /t 10

REM 执行 loop_check.bat
call loop_check.bat

echo Script completed.
pause
