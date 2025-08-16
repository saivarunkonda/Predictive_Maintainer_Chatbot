#!/usr/bin/env python3
"""
Setup script for Standalone RAG Application
"""
import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("🔧 Installing required packages...")
    
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'standalone_requirements.txt'])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def check_ollama():
    """Check if Ollama is accessible"""
    print("🔍 Checking Ollama connection...")
    
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json()
            available_models = [m['name'] for m in models['models']]
            print("✅ Ollama is running!")
            print(f"📋 Available models: {available_models}")
            
            if 'llama3.2:3b' in available_models:
                print("✅ llama3.2:3b model is available!")
            else:
                print("⚠️ llama3.2:3b model not found. You may need to pull it:")
                print("Run: ollama pull llama3.2:3b")
            
            return True
        else:
            print(f"❌ Ollama returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        print("Please ensure Ollama is running: ollama serve")
        return False

def create_test_files():
    """Create sample test files"""
    print("📝 Creating sample test files...")
    
    try:
        import pandas as pd
        
        # Create sample data
        data = {
            'Product': ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Speakers'],
            'Price': [999.99, 25.50, 75.00, 299.99, 150.00],
            'Category': ['Electronics', 'Accessories', 'Accessories', 'Electronics', 'Electronics'],
            'Stock': [50, 200, 100, 75, 30],
            'Rating': [4.5, 4.2, 4.7, 4.3, 4.1]
        }
        
        df = pd.DataFrame(data)
        
        # Create test directory
        if not os.path.exists('test_files'):
            os.makedirs('test_files')
        
        # Save test files
        df.to_excel('test_files/sample_products.xlsx', index=False)
        df.to_csv('test_files/sample_employees.csv', index=False)
        
        # Create text file
        with open('test_files/company_info.txt', 'w') as f:
            f.write("""Company Information

Our company specializes in technology products and accessories.

Product Categories:
- Electronics: Laptops, Monitors, Speakers
- Accessories: Mouse, Keyboard

We maintain high-quality standards with customer ratings above 4.0.
Contact us for bulk orders and special pricing.

Support: support@company.com
Sales: sales@company.com
""")
        
        print("✅ Sample test files created in 'test_files' directory")
        return True
        
    except Exception as e:
        print(f"❌ Error creating test files: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Setting up Standalone RAG Application")
    print("=" * 50)
    
    # Install requirements
    if not install_requirements():
        return
    
    # Check Ollama
    ollama_ok = check_ollama()
    
    # Create test files
    test_files_ok = create_test_files()
    
    print("\n" + "=" * 50)
    print("📋 SETUP SUMMARY:")
    print(f"Requirements: ✅ INSTALLED")
    print(f"Ollama: {'✅ READY' if ollama_ok else '❌ NEEDS SETUP'}")
    print(f"Test Files: {'✅ CREATED' if test_files_ok else '❌ FAILED'}")
    
    if ollama_ok and test_files_ok:
        print("\n🎉 Setup complete! Ready to run the application.")
        print("\nTo start the application:")
        print("streamlit run standalone_rag_app.py")
        print("\nTest files are available in the 'test_files' directory.")
    else:
        print("\n⚠️ Setup completed with some issues.")
        if not ollama_ok:
            print("Please ensure Ollama is running and pull the required model:")
            print("ollama pull llama3.2:3b")

if __name__ == "__main__":
    main()
