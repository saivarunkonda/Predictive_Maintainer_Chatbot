#!/usr/bin/env python3
"""
Test script for RAG application with Ollama
"""
import os
import sys
import requests
from pathlib import Path

def test_ollama_connection():
    """Test if Ollama is running and accessible"""
    print("🔍 Testing Ollama connection...")
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json()
            print("✅ Ollama is running!")
            print(f"📋 Available models: {[m['name'] for m in models['models']]}")
            return True, models['models']
        else:
            print(f"❌ Ollama responded with status {response.status_code}")
            return False, []
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        return False, []

def test_model_availability():
    """Test if llama3.2:3b model is available"""
    print("\n🔍 Checking model availability...")
    is_connected, models = test_ollama_connection()
    
    if not is_connected:
        return False
    
    model_names = [m['name'] for m in models]
    
    if 'llama3.2:3b' in model_names:
        print("✅ llama3.2:3b model is available!")
        return True
    else:
        print("⚠️ llama3.2:3b model not found")
        if model_names:
            print(f"Available models: {model_names}")
        else:
            print("No models found. You may need to pull a model first:")
            print("Run: ollama pull llama3.2:3b")
        return False

def test_simple_chat():
    """Test a simple chat with Ollama"""
    print("\n🔍 Testing simple chat with Ollama...")
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2:3b",
                "prompt": "Hello! Please respond with just 'Hello, I am working!' to confirm you're functioning.",
                "stream": False
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Model response: {result.get('response', 'No response')}")
            return True
        else:
            print(f"❌ Chat test failed with status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Chat test failed: {e}")
        return False

def check_streamlit_app():
    """Check if Streamlit app files exist"""
    print("\n🔍 Checking Streamlit app structure...")
    
    required_files = [
        "utils.py",
        "streaming.py",
        "secrets.toml",
        "pages/4_📄_chat_with_your_documents.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All required files found!")
        return True

def main():
    """Main test function"""
    print("🚀 Testing RAG Application Setup")
    print("=" * 50)
    
    # Test Ollama connection
    ollama_ok = test_ollama_connection()
    
    # Test model availability
    model_ok = test_model_availability()
    
    # Test simple chat (only if model is available)
    if model_ok:
        chat_ok = test_simple_chat()
    else:
        chat_ok = False
        print("⏳ Skipping chat test - model not available")
    
    # Check Streamlit app
    app_ok = check_streamlit_app()
    
    print("\n" + "=" * 50)
    print("📋 TEST RESULTS:")
    print(f"Ollama Connection: {'✅ PASS' if ollama_ok else '❌ FAIL'}")
    print(f"Model Available: {'✅ PASS' if model_ok else '❌ FAIL'}")
    print(f"Chat Test: {'✅ PASS' if chat_ok else '❌ FAIL'}")
    print(f"App Structure: {'✅ PASS' if app_ok else '❌ FAIL'}")
    
    if all([ollama_ok, model_ok, chat_ok, app_ok]):
        print("\n🎉 All tests passed! Your RAG application is ready!")
        print("\nTo run the application:")
        print("streamlit run Home.py")
        print("\nThen navigate to: 📄 Chat with your documents (Basic RAG)")
    else:
        print("\n⚠️ Some tests failed. Please check the issues above.")
        if not model_ok:
            print("\n💡 To fix model issues:")
            print("1. Make sure Ollama is running")
            print("2. Pull the model: ollama pull llama3.2:3b")
            print("3. Wait for the download to complete")

if __name__ == "__main__":
    main()
