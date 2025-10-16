# quick_test.ps1 - Fast API Health Check

param([string]$Url = "https://fruity-mails-join.loca.lt")

Write-Host "`n🚀 Quick Health Check" -ForegroundColor Cyan

$tests = @(
    @{ Name = "API Root"; Endpoint = "/" },
    @{ Name = "Health"; Endpoint = "/health" },
    @{ Name = "Model Status"; Endpoint = "/model/status" },
    @{ Name = "Model Info"; Endpoint = "/model/info" }
)

$passed = 0

foreach ($test in $tests) {
    try {
        $response = Invoke-RestMethod -Uri "$Url$($test.Endpoint)" -TimeoutSec 5 -ErrorAction Stop
        Write-Host "✅ $($test.Name)" -ForegroundColor Green
        $passed++
    } catch {
        Write-Host "❌ $($test.Name)" -ForegroundColor Red
    }
}

Write-Host "`n$passed/$($tests.Count) tests passed" -ForegroundColor $(if ($passed -eq $tests.Count) {"Green"} else {"Yellow"})