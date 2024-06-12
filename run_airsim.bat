@echo off

CALL C:\ProgramData\miniconda3\Scripts\activate.bat C:\Users\Administrator\.conda\envs\client

cd C:\Users\Administrator\Documents\UAV\Single_Agent_minigrid

python airsim_client\airsim_runner.py -f "%~1"

echo.
echo Task completed with config: %~1
echo Press any key to close this window...
pause > nul
