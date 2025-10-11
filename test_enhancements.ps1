# test_enhancements.ps1
Write-Host "🧪 Testing All Phase 2A Enhancements" -ForegroundColor Green

# Test 1: Orchestrator Health
Write-Host "`n=== Test 1: Orchestrator Health ===" -ForegroundColor Cyan
curl http://localhost:8004/health | ConvertTo-Json

# Test 2: Workflow Types
Write-Host "`n=== Test 2: Available Workflows ===" -ForegroundColor Cyan
curl http://localhost:8004/workflow/types | ConvertTo-Json

# Test 3: Quick Check Workflow
Write-Host "`n=== Test 3: Quick Check Workflow ===" -ForegroundColor Cyan
$body = @{
    workflow_type = "quick_check"
    locations = @("LOC001", "LOC002")
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8004/workflow/execute" `
    -Method Post `
    -ContentType "application/json" `
    -Body $body | ConvertTo-Json -Depth 5

# Test 4: Enhanced Predictions
Write-Host "`n=== Test 4: Enhanced Predictions ===" -ForegroundColor Cyan
$predBody = @{
    location_codes = @("LOC001")
    forecast_days = 7
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8002/predict" `
    -Method Post `
    -ContentType "application/json" `
    -Body $predBody | ConvertTo-Json -Depth 10

# Test 5: Alert System
Write-Host "`n=== Test 5: Alert Generation ===" -ForegroundColor Cyan
curl -X POST http://localhost:8000/alerts/generate | ConvertTo-Json -Depth 5

# Test 6: Alert History
Write-Host "`n=== Test 6: Alert History ===" -ForegroundColor Cyan
curl http://localhost:8003/alerts/history | ConvertTo-Json

# Test 7: System Statistics
Write-Host "`n=== Test 7: System Statistics ===" -ForegroundColor Cyan
curl http://localhost:8002/system/statistics | ConvertTo-Json

# Test 8: Full Analysis Workflow
Write-Host "`n=== Test 8: Full Analysis Workflow ===" -ForegroundColor Cyan
$fullBody = @{
    workflow_type = "full_analysis"
    locations = @("LOC001", "LOC002", "LOC003")
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8004/workflow/execute" `
    -Method Post `
    -ContentType "application/json" `
    -Body $fullBody | ConvertTo-Json -Depth 8

Write-Host "`n✅ All tests completed!" -ForegroundColor Green