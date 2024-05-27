@echo off
setlocal

set AIRSIM_PATH=c:\Users\Administrator\Documents\airsimcar41453\AirSimAssets.exe
set AIRSIM_SETTING=c:\Users\Administrator\Documents\AirSim\settings.json

echo Starting AirSim from %AIRSIM_PATH% with settings %AIRSIM_SETTING%
"%AIRSIM_PATH%" -settings="%AIRSIM_SETTING%"

if %ERRORLEVEL% neq 0 (
    echo Failed to start AirSim: Error %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

endlocal
