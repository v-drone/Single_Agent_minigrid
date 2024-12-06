# kill_airsim.ps1

param (
    [int]$unityPID,
    [string]$dockerContainerName
)

# Terminate Unity process
try {
    Stop-Process -Id $unityPID -Force
    Write-Host "Unity process with PID $unityPID has been terminated."
} catch {
    Write-Host "Failed to terminate Unity process with PID ${unityPID}: $_"
}

# Stop Docker container
try {
    docker stop $dockerContainerName
    Write-Host "Docker container $dockerContainerName has been stopped."
} catch {
    Write-Host "Failed to stop Docker container ${dockerContainerName}: $_"
}

# Terminate any related Python processes
try {
    Get-Process -Name "python" | Where-Object { $_.Path -like "*airsim_car_runner.py*" } | ForEach-Object { Stop-Process -Id $_.Id -Force }
    Write-Host "Related Python processes have been terminated."
} catch {
    Write-Host "Failed to terminate Python processes related to AirSim: $_"
}
