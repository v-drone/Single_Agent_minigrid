@echo off
REM 遍历airsim_configs目录下的所有配置文件
for %%f in (C:\Users\Administrator\Documents\UAV\Single_Agent_minigrid\airsim_configs\*.json) do (
    REM 在新的Windows Terminal标签中启动run_airsim.bat，传递配置文件路径
    start wt new-tab --title "Run AirSim with %%~nxf" cmd /c "C:\Users\Administrator\Documents\UAV\Single_Agent_minigrid\run_airsim.bat %%f"
)

echo All AirSim instances have been started.
pause
