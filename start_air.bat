@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

REM Setting up the path to the Conda activation script and the target environment
set "ACTIVATE_CMD=C:\ProgramData\miniconda3\Scripts\activate.bat C:\Users\Administrator\.conda\envs\client"

REM Setting the directory to change to
set "PROJECT_DIR=C:\Users\Administrator\Documents\UAV\Single_Agent_minigrid"

REM Initialize the command to open Windows Terminal
set "START_CMD=wt"

REM Loop through all JSON configuration files in the specified directory
for %%f in (!PROJECT_DIR!\airsim_configs\*.json) do (
    REM Append a new tab command for each config file
    set "START_CMD=!START_CMD! new-tab --title Run %%~nf cmd /k "!ACTIVATE_CMD! && cd !PROJECT_DIR! && python airsim_client\airsim_runner.py -f %%f && echo. && echo Task completed in %%~nf. Press any key to exit... && pause>nul"
)

REM Print the final command for debugging
echo !START_CMD!

REM Uncomment the line below to execute the command after verifying
REM !START_CMD!

echo All commands have been built.
pause

ENDLOCAL
