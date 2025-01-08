# kill_airsim.ps1

param (
    [int]$unityPID,
)

# Terminate Unity process
try {
    Stop-Process -Id $unityPID -Force
    Write-Host "Unity process with PID $unityPID has been terminated."
} catch {
    Write-Host "Failed to terminate Unity process with PID ${unityPID}: $_"
}


# Terminate any related Python processes
try {
    Get-Process -Name "python" | Where-Object { $_.Path -like "*airsim_car_runner.py*" } | ForEach-Object { Stop-Process -Id $_.Id -Force }
    Write-Host "Related Python processes have been terminated."
} catch {
    Write-Host "Failed to terminate Python processes related to AirSim: $_"
}
