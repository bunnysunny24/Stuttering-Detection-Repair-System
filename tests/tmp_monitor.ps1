$cpu = (Get-Counter '\Processor(_Total)\% Processor Time' -SampleInterval 1 -MaxSamples 1).CounterSamples.CookedValue
$os = Get-CimInstance Win32_OperatingSystem
$freeMB = [math]::Round($os.FreePhysicalMemory/1024,2)
$totalMB = [math]::Round($os.TotalVisibleMemorySize/1024,2)
Write-Output "CPU%: $([math]::Round($cpu,2))"
Write-Output "FreeMB: $freeMB TotalMB: $totalMB"
Get-Process -Name python -ErrorAction SilentlyContinue | Sort-Object -Property WS -Descending | Select-Object -First 10 Name,Id,CPU,@{Name='RSS_MB';Expression={[math]::Round($_.WS/1MB,2)}} | Format-Table -AutoSize
