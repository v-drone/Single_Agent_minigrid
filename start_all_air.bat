@echo off
CALL C:\ProgramData\miniconda3\Scripts\activate.bat C:\Users\Administrator\.conda\envs\client

REM 遍历airsim_configs目录下特定的配置文件
for %%i in (5000 5001 5002 5003 5004 5005 5006 5007) do (
    REM 在新的Windows Terminal标签中启动run_airsim.bat，传递配置文件路径
    start wt new-tab --title "Run AirSim with %%i" cmd /c "C:\Users\Administrator\Documents\UAV\Single_Agent_minigrid\run_airsim.bat C:\Users\Administrator\Documents\UAV\Single_Agent_minigrid\airsim_configs\%%i.json"
)

echo All specified AirSim instances have been started.
pause
