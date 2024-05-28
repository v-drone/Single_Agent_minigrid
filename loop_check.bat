@echo off
CALL C:\ProgramData\miniconda3\Scripts\activate.bat C:\Users\Administrator\.conda\envs\client

:loop
REM 运行Python脚本更新端口信息
python manage_services.py

REM 读取待启动的端口列表
for /f "delims=" %%a in ('type server_ports.json ^| python -c "import sys, json; print('\n'.join(str(port) for port in json.load(sys.stdin).get('todo_ports', [])))"') do (
    REM 启动新的服务实例
    echo Starting server on port %%a...
    start wt new-tab --title "Run AirSim with %%a" cmd /c "C:\path\to\airsim_configs\run_airsim.bat C:\path\to\airsim_configs\%%a.json"
    REM 添加端口到管理服务池，确保服务启动成功后执行
    python -c "import requests; requests.post('http://127.0.0.1:7575/add', json={'port': %%a})"
)

REM 清空todo_ports列表
python -c "import json; data = json.load(open('server_ports.json')); data['todo_ports'] = []; json.dump(data, open('server_ports.json', 'w'), indent=4)"

REM 等待一段时间再次执行
timeout /t 30
goto loop
