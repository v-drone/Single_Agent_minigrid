# PowerShell script

# Set error preference to silently continue on error
$ErrorActionPreference = 'SilentlyContinue'

# Set the port number from the script's argument
$port = $args[0]

# Define configuration and log paths based on the port
$configPath = "C:\Users\Administrator\Documents\UAV\Single_Agent_minigrid\airsim_configs\$port.json"
$logPath = "C:\Users\Administrator\Documents\UAV\Single_Agent_minigrid\Logs\$port.log"

# Calculate the Unity port
$unityPortOffset = 36450
$unityPort = $port + $unityPortOffset

# Paths for Unity executable and settings
$unityExecutablePath = "C:\Users\Administrator\Documents\airsimcar$unityPort\AirSimAssets.exe"
$settingsPath = "C:\Users\Administrator\Documents\AirSim\settings.json"
$mapPath = "C:\Users\Administrator\Documents\airsimcar$unityPort\AirSimAssets_Data\StreamingAssets\Test1.json"

# Verify Unity executable's existence
if (Test-Path -Path $unityExecutablePath -PathType Leaf)
{
    try
    {
        # Start Unity
        $command = "`"$unityExecutablePath`" `"$settingsPath`""
        Write-Host "Executing Unity command: $command"
        $process = Start-Process -FilePath $unityExecutablePath -ArgumentList $settingsPath -PassThru
        Start-Sleep -Seconds 10

        # Verify Unity process is running
        if ($process -and !$process.HasExited)
        {
            Write-Host "Unity started successfully on port $unityPort"
            $unityPID = $process.Id
            Write-Host "Unity started with PID: $unityPID"

            # Set the directory for subsequent operations
            Set-Location -Path "C:\Users\Administrator\Documents\UAV\Single_Agent_minigrid"

            # Start Docker container
            Write-Host "Starting Docker container for port $port..."
            docker start "$port"

            # Define Python command
            $pythonScriptPath = ".\airsim_client\airsim_runner.py"
            $pythonCommand = "python $pythonScriptPath -f $configPath -l $logPath -p $unityPID"
            Write-Host "Executing Python command: $pythonCommand"

            # Execute Python script with all necessary parameters
            Invoke-Expression $pythonCommand

            # Completion message
            Write-Host "`nTask completed with config: $configPath"
            Write-Host "Press any key to close this window..."
            $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

        }
        else
        {
            throw "Unity process terminated prematurely."
        }
    }
    catch
    {
        Write-Host "Failed to start Unity: $_"
    }
}
else
{
    Write-Host "Unity executable not found or not executable at path: $unityExecutablePath"
}

# This script now thoroughly integrates the processes for a streamlined execution.
