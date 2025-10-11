# demo_script.ps1

# Clear the screen
Clear-Host

# Display the header using Write-Host for color
Write-Host "═══════════════════════════════════════" -ForegroundColor Cyan
Write-Host "   EpiSPY - Live Demo" -ForegroundColor White
Write-Host "   Epidemic Surveillance System" -ForegroundColor White
Write-Host "═══════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# Step 1: Health Check
Write-Host "📊 Step 1: Health Check" -ForegroundColor Green
try {
    $health = Invoke-RestMethod -Uri "http://localhost:8000/health"
    # Format the output to look nice
    $health | Select-Object overall, @{Name="services"; Expression={$_.services.PSObject.Properties.Name}} | Format-List
} catch {
    Write-Warning "Could not connect to the server. Is it running?"
}
Start-Sleep -Seconds 2

# Step 2: Data Ingestion
Write-Host "📥 Step 2: Data Ingestion" -ForegroundColor Green
try {
    Invoke-RestMethod -Method Post -Uri "http://localhost:8000/data/ingest" | Select-Object status, records_processed | Format-List
} catch {
    Write-Warning "Data ingestion failed."
}
Start-Sleep -Seconds 2

# Step 3: Epidemic Forecast (7 days)
Write-Host "🔮 Step 3: Epidemic Forecast (7 days)" -ForegroundColor Green
try {
    # Create the request body AND convert it to a JSON string
    $forecastBody = @{
        locations = @("LOC001", "LOC002")
    } | ConvertTo-Json # <-- THIS LINE IS THE FIX

    $forecast = Invoke-RestMethod -Method Post -Uri "http://localhost:8000/prediction/forecast" -Body $forecastBody -ContentType 'application/json'
    
    # Select the first result and rename properties to match the jq output
    $forecast[0] | Select-Object @{N='location'; E={$_.location_code}}, @{N='current'; E={$_.current_cases}}, @{N='forecast'; E={$_.predicted_cases}}, trend | Format-List
} catch {
    Write-Warning "Forecast prediction failed."
}
Start-Sleep -Seconds 3

# Step 4: AI Alert Generation (LLM)
Write-Host "🤖 Step 4: AI Alert Generation (LLM)" -ForegroundColor Green
try {
    # Create the request body AND convert it to a JSON string
    $alertBody = @{
        locations = @("LOC001", "LOC002")
    } | ConvertTo-Json # <-- THIS LINE IS THE FIX

    $alert = Invoke-RestMethod -Method Post -Uri "http://localhost:8000/alerts/generate" -Body $alertBody -ContentType 'application/json'
    
    # Select properties and the first two recommendations
    $alert | Select-Object severity, summary, @{N='recommendations'; E={$_.recommendations[0..1]}} | Format-List
} catch {
    Write-Warning "Alert generation failed."
}
Write-Host ""
Write-Host "✅ Demo Complete!" -ForegroundColor Green