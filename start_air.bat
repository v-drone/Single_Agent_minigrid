@echo off
REM 激活Conda环境的命令
set ACTIVATE_CMD="C:\ProgramData\miniconda3\Scripts\activate.bat C:\Users\Administrator\.conda\envs\client"

REM 进入项目目录并设置Python脚本运行命令
set PYTHON_CMD="cd C:\Users\Administrator\Documents\UAV\Single_Agent_minigrid && python airsim_client\airsim_runner.py"

REM 构造启动命令
set START_CMD=

REM 循环遍历airsim_configs目录下的所有配置文件
for %%f in (C:\Users\Administrator\Documents\UAV\Single_Agent_minigrid\airsim_configs\*.json) do (
    if defined START_CMD (
        set START_CMD=%START_CMD% ; -w 0 new-tab --title "Run %%~nf" cmd /k %ACTIVATE_CMD% && %PYTHON_CMD% "%%f" && echo. && echo Task completed in %%~nf. Press any key to exit... && pause>nul
    ) else (
        set START_SERIAL=-w 0 new-tab --title "Run %%~nf" cmd /k %ACTIVATE_CMD% && %PYTHON_CMD% "%%f" && echo. && echo Task completed in %%~nf. Press any key to exit... && pause>nul
    )
)

REM 执行Windows Terminal命令
wt %START_CMD%

echo All tasks have been started.
pause
