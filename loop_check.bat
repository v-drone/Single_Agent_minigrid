@echo off
:loop
REM 运行Python脚本更新端口信息
python manage_services.py

REM 读取待启动的端口列表
for /f "delims=" %%a in ('type server_ports.json ^| python -c "import sys, json; print('\n'.join(str(port) for port in json.load(sys.stdin).get('todo_ports', [])))"') do (
    REM 启动新的服务实例，确保路径指向airsim_configs文件夹
    echo Starting server on port %%a...
    start wt new-tab --title "Run AirSim with %%a" cmd /c "C:\Users\Administrator\Documents\UAV\Single_Agent_minigrid\run_airsim.bat C:\Users\Administrator\Documents\UAV\Single_Agent_minigrid\airsim_configs\%%a.json"
    REM 向管理服务池添加端口
    curl -X POST http://127.0.0.1:7575/add -d "{\"port\": %%a}"
)

REM 等待30秒再次执行
timeout /t 30
goto loop
