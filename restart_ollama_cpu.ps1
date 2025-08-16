# PowerShell script to restart Ollama in CPU mode
Write-Host "Stopping Ollama service..." -ForegroundColor Yellow
Get-Process -Name "ollama" -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 2

Write-Host "Setting CPU-only environment variables..." -ForegroundColor Yellow
$env:OLLAMA_NUM_PARALLEL = "1"
$env:OLLAMA_MAX_LOADED_MODELS = "1"
$env:OLLAMA_NUM_GPU = "0"
$env:OLLAMA_LLM_LIBRARY = "cpu"

Write-Host "Starting Ollama in CPU mode..." -ForegroundColor Yellow
Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden

Write-Host "Waiting for Ollama to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

Write-Host "Testing Ollama connection..." -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method Get
    Write-Host "✅ Ollama is running successfully!" -ForegroundColor Green
    Write-Host "Available models:" -ForegroundColor Cyan
    $response.models | ForEach-Object { Write-Host "  - $($_.name)" -ForegroundColor White }
} catch {
    Write-Host "❌ Failed to connect to Ollama: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "`nOllama is now configured for CPU-only mode." -ForegroundColor Green
Write-Host "You can now run your RAG application with: streamlit run Home.py" -ForegroundColor Cyan
