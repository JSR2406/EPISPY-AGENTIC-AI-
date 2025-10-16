# test_enhancements.ps1
# Enhanced Testing Suite for EpiSPY Phase 2A
# ═══════════════════════════════════════════════

param(
    [string]$BaseUrl = "https://fruity-mails-join.loca.lt",
    [switch]$Verbose,
    [switch]$StopOnError
)

# Initialize counters
$script:PassedTests = 0
$script:FailedTests = 0
$script:TotalTests = 0
$script:TestResults = @()

# Color scheme
$Colors = @{
    Header = "Cyan"
    Success = "Green"
    Error = "Red"
    Warning = "Yellow"
    Info = "White"
    Highlight = "Magenta"
}

# Test result class
class TestResult {
    [string]$TestName
    [string]$Status
    [double]$Duration
    [string]$Message
    [object]$Response
}

function Write-TestHeader {
    param([string]$Message)
    Write-Host "`n" -NoNewline
    Write-Host "═══════════════════════════════════════════════" -ForegroundColor $Colors.Header
    Write-Host "  $Message" -ForegroundColor $Colors.Header
    Write-Host "═══════════════════════════════════════════════" -ForegroundColor $Colors.Header
}

function Write-TestResult {
    param(
        [string]$TestName,
        [bool]$Passed,
        [string]$Message = "",
        [double]$Duration = 0
    )
    
    $script:TotalTests++
    
    if ($Passed) {
        $script:PassedTests++
        Write-Host "✅ " -NoNewline -ForegroundColor $Colors.Success
        Write-Host "$TestName " -NoNewline -ForegroundColor $Colors.Info
        Write-Host "($([math]::Round($Duration, 2))ms)" -ForegroundColor $Colors.Info
        if ($Message) {
            Write-Host "   ℹ️  $Message" -ForegroundColor $Colors.Info
        }
    } else {
        $script:FailedTests++
        Write-Host "❌ " -NoNewline -ForegroundColor $Colors.Error
        Write-Host "$TestName " -NoNewline -ForegroundColor $Colors.Info
        Write-Host "FAILED" -ForegroundColor $Colors.Error
        if ($Message) {
            Write-Host "   ⚠️  $Message" -ForegroundColor $Colors.Warning
        }
    }
    
    # Store result
    $result = [TestResult]@{
        TestName = $TestName
        Status = if ($Passed) { "PASSED" } else { "FAILED" }
        Duration = $Duration
        Message = $Message
    }
    $script:TestResults += $result
}

function Invoke-ApiTest {
    param(
        [string]$TestName,
        [string]$Endpoint,
        [string]$Method = "GET",
        [object]$Body = $null,
        [scriptblock]$Validator = $null
    )
    
    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    
    try {
        $params = @{
            Uri = "$BaseUrl$Endpoint"
            Method = $Method
            ErrorAction = "Stop"
            TimeoutSec = 30
        }
        
        if ($Body) {
            $params.ContentType = "application/json"
            $params.Body = $Body | ConvertTo-Json -Depth 10
        }
        
        $response = Invoke-RestMethod @params
        $stopwatch.Stop()
        
        # Run custom validator if provided
        $validationPassed = $true
        $validationMessage = ""
        
        if ($Validator) {
            try {
                $validationResult = & $Validator $response
                if ($validationResult -is [hashtable]) {
                    $validationPassed = $validationResult.Passed
                    $validationMessage = $validationResult.Message
                } else {
                    $validationPassed = $validationResult
                }
            } catch {
                $validationPassed = $false
                $validationMessage = "Validation failed: $_"
            }
        }
        
        if ($validationPassed) {
            Write-TestResult -TestName $TestName -Passed $true -Duration $stopwatch.ElapsedMilliseconds -Message $validationMessage
            
            if ($Verbose) {
                Write-Host "   Response:" -ForegroundColor $Colors.Info
                $response | ConvertTo-Json -Depth 5 | Write-Host -ForegroundColor DarkGray
            }
        } else {
            Write-TestResult -TestName $TestName -Passed $false -Message $validationMessage
        }
        
        return @{ Success = $validationPassed; Response = $response; Duration = $stopwatch.ElapsedMilliseconds }
        
    } catch {
        $stopwatch.Stop()
        $errorMessage = $_.Exception.Message
        Write-TestResult -TestName $TestName -Passed $false -Message $errorMessage
        
        if ($StopOnError) {
            throw "Test failed: $TestName - $errorMessage"
        }
        
        return @{ Success = $false; Error = $errorMessage; Duration = $stopwatch.ElapsedMilliseconds }
    }
}

# ═══════════════════════════════════════════════
# MAIN TEST EXECUTION
# ═══════════════════════════════════════════════

Clear-Host

Write-Host "`n" -NoNewline
Write-Host "🧪 " -NoNewline -ForegroundColor $Colors.Highlight
Write-Host "EpiSPY Phase 2A Enhancement Testing Suite" -ForegroundColor $Colors.Highlight
Write-Host "═══════════════════════════════════════════════" -ForegroundColor $Colors.Highlight
Write-Host "📡 Target: " -NoNewline -ForegroundColor $Colors.Info
Write-Host "$BaseUrl" -ForegroundColor $Colors.Warning
Write-Host "⏰ Started: " -NoNewline -ForegroundColor $Colors.Info
Write-Host "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor $Colors.Info
Write-Host ""

$overallStopwatch = [System.Diagnostics.Stopwatch]::StartNew()

# ═══════════════════════════════════════════════
# CATEGORY 1: SYSTEM HEALTH CHECKS
# ═══════════════════════════════════════════════

Write-TestHeader "CATEGORY 1: System Health Checks"

Invoke-ApiTest `
    -TestName "Root Endpoint" `
    -Endpoint "/" `
    -Validator {
        param($response)
        return @{
            Passed = $response.status -eq "running"
            Message = "API Version: $($response.version)"
        }
    }

Start-Sleep -Milliseconds 500

Invoke-ApiTest `
    -TestName "Health Check" `
    -Endpoint "/health" `
    -Validator {
        param($response)
        $modelStatus = $response.model_info.status
        return @{
            Passed = $response.overall -eq "healthy" -or $response.overall -eq "degraded"
            Message = "Overall: $($response.overall) | Model: $modelStatus"
        }
    }

Start-Sleep -Milliseconds 500

Invoke-ApiTest `
    -TestName "Model Status" `
    -Endpoint "/model/status" `
    -Validator {
        param($response)
        return @{
            Passed = $response.status -ne $null
            Message = "Status: $($response.status) | Version: $($response.version)"
        }
    }

# ═══════════════════════════════════════════════
# CATEGORY 2: MODEL DEPLOYMENT & INFO
# ═══════════════════════════════════════════════

Write-TestHeader "CATEGORY 2: Model Deployment & Information"

Invoke-ApiTest `
    -TestName "Model Info" `
    -Endpoint "/model/info" `
    -Validator {
        param($response)
        $featureCount = $response.features.Count
        return @{
            Passed = $response.model_name -and $featureCount -gt 0
            Message = "Model: $($response.model_name) | Features: $featureCount"
        }
    }

Start-Sleep -Milliseconds 500

Invoke-ApiTest `
    -TestName "Model Health" `
    -Endpoint "/model/health" `
    -Validator {
        param($response)
        $uptime = [math]::Round($response.uptime_seconds / 60, 1)
        return @{
            Passed = $response.model_loaded -eq $true
            Message = "Loaded: $($response.model_loaded) | Uptime: ${uptime}m | Predictions: $($response.total_predictions)"
        }
    }

Start-Sleep -Milliseconds 500

Invoke-ApiTest `
    -TestName "Model Metrics" `
    -Endpoint "/model/metrics" `
    -Validator {
        param($response)
        $accuracy = [math]::Round($response.accuracy * 100, 1)
        return @{
            Passed = $response.accuracy -gt 0
            Message = "Accuracy: ${accuracy}% | Total Predictions: $($response.total_predictions_today)"
        }
    }

# ═══════════════════════════════════════════════
# CATEGORY 3: PREDICTION ENDPOINTS
# ═══════════════════════════════════════════════

Write-TestHeader "CATEGORY 3: Prediction Capabilities"

$singlePredictionBody = @{
    location_id = "LOC001"
    cases = 75
    severity_score = 6.8
    population_density = 5500
    weather_temp = 29.5
}

Invoke-ApiTest `
    -TestName "Single Prediction" `
    -Endpoint "/model/predict" `
    -Method "POST" `
    -Body $singlePredictionBody `
    -Validator {
        param($response)
        return @{
            Passed = $response.risk_level -and $response.confidence
            Message = "Risk: $($response.risk_level) | Confidence: $([math]::Round($response.confidence * 100, 1))% | Score: $([math]::Round($response.risk_score, 1))"
        }
    }

Start-Sleep -Milliseconds 500

$batchPredictionBody = @{
    predictions = @(
        @{ location_id = "LOC001"; cases = 45; severity_score = 5.5; population_density = 4500; weather_temp = 28 },
        @{ location_id = "LOC002"; cases = 120; severity_score = 8.2; population_density = 7000; weather_temp = 31 },
        @{ location_id = "LOC003"; cases = 28; severity_score = 3.1; population_density = 2500; weather_temp = 26 },
        @{ location_id = "LOC004"; cases = 95; severity_score = 7.5; population_density = 6200; weather_temp = 30 }
    )
}

Invoke-ApiTest `
    -TestName "Batch Prediction" `
    -Endpoint "/model/predict/batch" `
    -Method "POST" `
    -Body $batchPredictionBody `
    -Validator {
        param($response)
        $processTime = [math]::Round($response.processing_time_ms, 1)
        return @{
            Passed = $response.total_processed -eq 4
            Message = "Processed: $($response.total_processed) locations | Time: ${processTime}ms"
        }
    }

Start-Sleep -Milliseconds 500

$forecastBody = @{
    locations = @("LOC001", "LOC002", "LOC003")
}

Invoke-ApiTest `
    -TestName "7-Day Forecast" `
    -Endpoint "/prediction/forecast" `
    -Method "POST" `
    -Body $forecastBody `
    -Validator {
        param($response)
        $locationCount = $response.Count
        return @{
            Passed = $locationCount -eq 3
            Message = "Forecasts generated for $locationCount locations"
        }
    }

# ═══════════════════════════════════════════════
# CATEGORY 4: ALERT SYSTEM
# ═══════════════════════════════════════════════

Write-TestHeader "CATEGORY 4: Alert Generation & Management"

$alertBody = @{
    locations = @("LOC001", "LOC002", "LOC003")
}

Invoke-ApiTest `
    -TestName "Alert Generation" `
    -Endpoint "/alerts/generate" `
    -Method "POST" `
    -Body $alertBody `
    -Validator {
        param($response)
        return @{
            Passed = $response.severity -and $response.alert_id
            Message = "Alert ID: $($response.alert_id) | Severity: $($response.severity)"
        }
    }

# ═══════════════════════════════════════════════
# CATEGORY 5: DATA MANAGEMENT
# ═══════════════════════════════════════════════

Write-TestHeader "CATEGORY 5: Data Management"

Invoke-ApiTest `
    -TestName "Data Ingestion" `
    -Endpoint "/data/ingest" `
    -Method "POST" `
    -Validator {
        param($response)
        return @{
            Passed = $response.status -eq "success"
            Message = "Records processed: $($response.records_processed) | Time: $([math]::Round($response.processing_time_ms, 1))ms"
        }
    }

# ═══════════════════════════════════════════════
# CATEGORY 6: WORKFLOW EXECUTION
# ═══════════════════════════════════════════════

Write-TestHeader "CATEGORY 6: Workflow Execution"

$quickCheckBody = @{
    workflow_type = "quick_check"
    locations = @("LOC001", "LOC002")
}

Invoke-ApiTest `
    -TestName "Quick Check Workflow" `
    -Endpoint "/workflow/execute" `
    -Method "POST" `
    -Body $quickCheckBody `
    -Validator {
        param($response)
        return @{
            Passed = $response.status -eq "completed"
            Message = "ID: $($response.workflow_id) | Locations: $($response.locations_processed)"
        }
    }

Start-Sleep -Milliseconds 500

$fullAnalysisBody = @{
    workflow_type = "full_analysis"
    locations = @("LOC001", "LOC002", "LOC003", "LOC004", "LOC005")
}

Invoke-ApiTest `
    -TestName "Full Analysis Workflow" `
    -Endpoint "/workflow/execute" `
    -Method "POST" `
    -Body $fullAnalysisBody `
    -Validator {
        param($response)
        $totalTime = [math]::Round($response.total_duration_ms, 1)
        return @{
            Passed = $response.status -eq "completed"
            Message = "Steps: $($response.steps_completed.Count) | Total Time: ${totalTime}ms"
        }
    }

# ═══════════════════════════════════════════════
# CATEGORY 7: PERFORMANCE & STRESS TESTS
# ═══════════════════════════════════════════════

Write-TestHeader "CATEGORY 7: Performance Testing"

Write-Host "⚡ Running concurrent prediction tests..." -ForegroundColor $Colors.Warning

$concurrentTests = @()
$concurrentStopwatch = [System.Diagnostics.Stopwatch]::StartNew()

for ($i = 1; $i -le 5; $i++) {
    $testBody = @{
        location_id = "LOC00$i"
        cases = Get-Random -Minimum 20 -Maximum 150
        severity_score = [math]::Round((Get-Random -Minimum 1.0 -Maximum 10.0), 1)
        population_density = Get-Random -Minimum 1000 -Maximum 10000
        weather_temp = [math]::Round((Get-Random -Minimum 15.0 -Maximum 35.0), 1)
    }
    
    $job = Start-Job -ScriptBlock {
        param($url, $body)
        try {
            $response = Invoke-RestMethod -Uri "$url/model/predict" -Method Post -ContentType "application/json" -Body ($body | ConvertTo-Json) -TimeoutSec 30
            return @{ Success = $true; Response = $response }
        } catch {
            return @{ Success = $false; Error = $_.Exception.Message }
        }
    } -ArgumentList $BaseUrl, $testBody
    
    $concurrentTests += $job
}

$results = $concurrentTests | Wait-Job | Receive-Job
$concurrentStopwatch.Stop()

$successCount = ($results | Where-Object { $_.Success }).Count
$avgTime = [math]::Round($concurrentStopwatch.ElapsedMilliseconds / 5, 1)

Write-TestResult `
    -TestName "Concurrent Predictions (5x)" `
    -Passed ($successCount -eq 5) `
    -Duration $concurrentStopwatch.ElapsedMilliseconds `
    -Message "Success: $successCount/5 | Avg Time: ${avgTime}ms per request"

$concurrentTests | Remove-Job -Force

# ═══════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════

$overallStopwatch.Stop()

Write-Host "`n" -NoNewline
Write-Host "═══════════════════════════════════════════════" -ForegroundColor $Colors.Highlight
Write-Host "  📊 TEST SUMMARY" -ForegroundColor $Colors.Highlight
Write-Host "═══════════════════════════════════════════════" -ForegroundColor $Colors.Highlight

$passRate = if ($script:TotalTests -gt 0) { [math]::Round(($script:PassedTests / $script:TotalTests) * 100, 1) } else { 0 }
$totalTime = [math]::Round($overallStopwatch.Elapsed.TotalSeconds, 2)

Write-Host "`n📈 Results:" -ForegroundColor $Colors.Info
Write-Host "   Total Tests:  " -NoNewline -ForegroundColor $Colors.Info
Write-Host "$script:TotalTests" -ForegroundColor $Colors.Highlight

Write-Host "   ✅ Passed:    " -NoNewline -ForegroundColor $Colors.Info
Write-Host "$script:PassedTests" -ForegroundColor $Colors.Success

Write-Host "   ❌ Failed:    " -NoNewline -ForegroundColor $Colors.Info
Write-Host "$script:FailedTests" -ForegroundColor $(if ($script:FailedTests -gt 0) { $Colors.Error } else { $Colors.Info })

Write-Host "   📊 Pass Rate: " -NoNewline -ForegroundColor $Colors.Info
Write-Host "${passRate}%" -ForegroundColor $(if ($passRate -ge 90) { $Colors.Success } elseif ($passRate -ge 70) { $Colors.Warning } else { $Colors.Error })

Write-Host "   ⏱️  Duration:  " -NoNewline -ForegroundColor $Colors.Info
Write-Host "${totalTime}s" -ForegroundColor $Colors.Info

# Export results
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$reportPath = "test_results_$timestamp.json"

$report = @{
    timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    base_url = $BaseUrl
    summary = @{
        total_tests = $script:TotalTests
        passed = $script:PassedTests
        failed = $script:FailedTests
        pass_rate = $passRate
        duration_seconds = $totalTime
    }
    results = $script:TestResults
}

$report | ConvertTo-Json -Depth 10 | Out-File -FilePath $reportPath -Encoding UTF8

Write-Host "`n📄 Detailed report saved to: " -NoNewline -ForegroundColor $Colors.Info
Write-Host "$reportPath" -ForegroundColor $Colors.Warning

# Final status
Write-Host "`n" -NoNewline
if ($script:FailedTests -eq 0) {
    Write-Host "🎉 ALL TESTS PASSED! System is fully operational." -ForegroundColor $Colors.Success
} elseif ($passRate -ge 80) {
    Write-Host "⚠️  MOSTLY PASSED with some failures. Review failed tests." -ForegroundColor $Colors.Warning
} else {
    Write-Host "❌ MULTIPLE FAILURES detected. System needs attention." -ForegroundColor $Colors.Error
}

Write-Host "`n═══════════════════════════════════════════════`n" -ForegroundColor $Colors.Highlight

# Exit with appropriate code
exit $(if ($script:FailedTests -eq 0) { 0 } else { 1 })