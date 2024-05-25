@echo off
REM 设置激活Conda环境的命令
set ACTIVATE_CMD="C:\ProgramData\miniconda3\Scripts\activate.bat C:\Users\Administrator\.conda\envs\client"

REM 设置进入项目目录并运行Python脚本的命令
set PYTHON_CMD="cd C:\Users\Administrator\Documents\UAV\Single_Agent_minigrid && python airsim_client\airsim_runner.py"

REM 初始化命令字符串
set START_CMD=

REM 循环遍历airsim_configs目录下的所有配置文件
for %%f in (C:\Users\Administrator\Documents\UAV\Single_Agent_minigrid\airsim_configs\*.json) do (
    if defined START_CMD (
        set START_CMD=%START_CMD% ; wt -w 0 new-tab --title "Run %%~nf" cmd /k %ACTIVATE_CMD% && %PYTHON_CMD% "%%f" && echo. && echo Task completed in %%~nf. Press any key to exit... && pause>nul
    ) else (
        set START_SERIAL=wt -w 0 new-tab --title "Run %%~nf" cmd /k %ACTIVATE_CMD% && %PYTHON_CMD% "%%f" && echo. && echo Task completed in %%~nf. Press any key to exit... && pause>nul
    )
)

REM 打印最终构建的命令字符串
echo %START_CMD%

echo All commands have been built.
pause
