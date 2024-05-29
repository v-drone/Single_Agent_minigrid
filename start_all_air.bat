@echo off
CALL C:\ProgramData\miniconda3\Scripts\activate.bat C:\Users\Administrator\.conda\envs\client


REM 创建并覆盖 backup_ports.txt 文件
echo Writing to backup_ports.txt
(for /l %%j in (5000,1,5020) do echo %%j) > backup_ports.txt

echo Ports backup completed.

REM 等待10秒
timeout /t 10

REM 执行 loop_check.bat
call airsim_check.bat

echo Script completed.
pause
