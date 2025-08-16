#!/usr/bin/env python3
"""
Test script to configure Ollama for CPU usage and test the RAG application
"""
import os
import sys
import time
import requests
import subprocess
from pathlib import Path

def set_cpu_only_environment():
    """Set environment variables for CPU-only mode"""
    print("🔧 Setting CPU-only environment variables...")
    
    # Set environment variables for CPU mode
    os.environ['OLLAMA_NUM_GPU'] = '0'
    os.environ['OLLAMA_LLM_LIBRARY'] = 'cpu'
    os.environ['OLLAMA_NUM_PARALLEL'] = '1'
    os.environ['OLLAMA_MAX_LOADED_MODELS'] = '1'
    
    print("✅ Environment variables set:")
    print(f"   OLLAMA_NUM_GPU: {os.environ.get('OLLAMA_NUM_GPU', 'Not set')}")
    print(f"   OLLAMA_LLM_LIBRARY: {os.environ.get('OLLAMA_LLM_LIBRARY', 'Not set')}")

def test_ollama_with_cpu():
    """Test Ollama with CPU-only configuration"""
    print("\n🔍 Testing Ollama with CPU configuration...")
    
    # Wait a bit for Ollama to restart
    time.sleep(5)
    
    try:
        # Test basic connection
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json()
            print("✅ Ollama is running!")
            print(f"📋 Available models: {[m['name'] for m in models['models']]}")
        else:
            print(f"❌ Ollama connection failed with status {response.status_code}")
            return False
            
        # Test a simple generation with CPU-optimized parameters
        test_response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2:3b",
                "prompt": "Hello! Say 'CPU mode working' to confirm.",
                "stream": False,
                "options": {
                    "num_ctx": 1024,  # Smaller context window
                    "num_gpu": 0,     # Force CPU
                    "num_thread": 4   # Limit threads
                }
            },
            timeout=60
        )
        
        if test_response.status_code == 200:
            result = test_response.json()
            print(f"✅ CPU Test successful! Response: {result.get('response', 'No response')}")
            return True
        else:
            print(f"❌ CPU Test failed with status {test_response.status_code}")
            print(f"Error: {test_response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing Ollama: {e}")
        return False

def main():
    """Main function"""
    print("🚀 Configuring Ollama for CPU mode")
    print("=" * 50)
    
    # Set environment variables
    set_cpu_only_environment()
    
    # Test Ollama
    success = test_ollama_with_cpu()
    
    if success:
        print("\n🎉 Ollama is now configured for CPU mode!")
        print("You can now run your RAG application:")
        print("streamlit run Home.py")
    else:
        print("\n⚠️ Ollama CPU configuration failed.")
        print("Please try manually restarting Ollama or check system resources.")

if __name__ == "__main__":
    main()
