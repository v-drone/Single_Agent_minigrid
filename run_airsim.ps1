# PowerShell script

# Set error preference to silently continue on error
$ErrorActionPreference = 'SilentlyContinue'

# Set the port number from the script's argument
$port = $args[0]

# Navigate to the specified directory
Set-Location -Path "C:\Users\Administrator\Documents\UAV\Single_Agent_minigrid"

# Define configuration and log paths based on the port
$configPath = "C:\Users\Administrator\Documents\UAV\Single_Agent_minigrid\airsim_configs\$port.json"
$logPath = "C:\Users\Administrator\Documents\UAV\Single_Agent_minigrid\Logs\$port.log"

# Start Docker container for the specified port
Write-Host "Starting Docker container for port $port..."
docker start "$port"

# Run the Python script using the paths defined above
python .\airsim_client\airsim_runner.py -f $configPath -l $logPath