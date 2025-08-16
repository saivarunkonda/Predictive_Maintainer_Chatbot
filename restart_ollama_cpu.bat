@echo off
echo Stopping Ollama service...
taskkill /f /im ollama.exe 2>nul
timeout /t 2 /nobreak >nul

echo Setting CPU-only mode...
set OLLAMA_NUM_PARALLEL=1
set OLLAMA_MAX_LOADED_MODELS=1
set OLLAMA_NUM_GPU=0
set OLLAMA_LLM_LIBRARY=cpu

echo Starting Ollama in CPU mode...
start /b ollama serve

echo Waiting for Ollama to start...
timeout /t 5 /nobreak >nul

echo Testing Ollama connection...
curl -s http://localhost:11434/api/tags

echo.
echo Ollama is now running in CPU-only mode
echo You can now run your RAG application
