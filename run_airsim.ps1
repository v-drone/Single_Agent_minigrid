# PowerShell script equivalent

# Navigate to the specified directory
Set-Location -Path "C:\Users\Administrator\Documents\UAV\Single_Agent_minigrid"

# Run the Python script with the provided command line argument
python .\airsim_client\airsim_runner.py -f $args[0] -l $args[1]

# Output completion message with the configuration file used
Write-Host "`nTask completed with config: $($args[0])"
Write-Host "Press any key to close this window..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
