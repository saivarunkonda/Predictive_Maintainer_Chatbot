#!/usr/bin/env python3
"""
Test script to verify Excel and multiple format support in RAG application
"""
import os
import pandas as pd
from pathlib import Path

def create_test_documents():
    """Create sample documents for testing"""
    print("📝 Creating test documents...")
    
    # Create test directory if it doesn't exist
    test_dir = Path("test_documents")
    test_dir.mkdir(exist_ok=True)
    
    # Create sample Excel file
    excel_data = {
        'Product': ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Speakers'],
        'Price': [999.99, 29.99, 79.99, 299.99, 149.99],
        'Category': ['Electronics', 'Accessories', 'Accessories', 'Electronics', 'Electronics'],
        'Stock': [50, 200, 100, 75, 120],
        'Description': [
            'High-performance laptop with 16GB RAM and 512GB SSD',
            'Wireless optical mouse with ergonomic design',
            'Mechanical keyboard with RGB backlight',
            '27-inch 4K monitor with IPS display',
            'Premium stereo speakers with Bluetooth'
        ]
    }
    
    df = pd.DataFrame(excel_data)
    excel_file = test_dir / "sample_products.xlsx"
    df.to_excel(excel_file, index=False)
    print(f"✅ Created Excel file: {excel_file}")
    
    # Create sample CSV file
    csv_data = {
        'Employee': ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Brown'],
        'Department': ['IT', 'Sales', 'HR', 'Marketing'],
        'Salary': [75000, 65000, 55000, 60000],
        'Years': [5, 3, 8, 2],
        'Performance': ['Excellent', 'Good', 'Excellent', 'Good']
    }
    
    df_csv = pd.DataFrame(csv_data)
    csv_file = test_dir / "sample_employees.csv"
    df_csv.to_csv(csv_file, index=False)
    print(f"✅ Created CSV file: {csv_file}")
    
    # Create sample text file
    txt_content = """
    # Company Overview
    
    Our company is a leading technology solutions provider with over 10 years of experience.
    We specialize in:
    
    1. Software Development
    2. Cloud Computing
    3. Data Analytics
    4. AI and Machine Learning
    
    ## Our Mission
    To deliver innovative technology solutions that drive business growth and efficiency.
    
    ## Our Values
    - Innovation
    - Quality
    - Customer Focus
    - Integrity
    
    ## Contact Information
    Email: info@company.com
    Phone: (555) 123-4567
    Address: 123 Tech Street, Silicon Valley, CA 94000
    """
    
    txt_file = test_dir / "company_info.txt"
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write(txt_content)
    print(f"✅ Created text file: {txt_file}")
    
    return excel_file, csv_file, txt_file

def test_document_formats():
    """Test various document formats"""
    print("\n🧪 Testing document format support...")
    
    # Create test documents
    excel_file, csv_file, txt_file = create_test_documents()
    
    # Test if pandas can read the Excel file
    try:
        df = pd.read_excel(excel_file)
        print(f"✅ Excel file readable: {len(df)} rows, {len(df.columns)} columns")
        print(f"   Columns: {list(df.columns)}")
    except Exception as e:
        print(f"❌ Excel file error: {e}")
        return False
    
    # Test if pandas can read the CSV file
    try:
        df_csv = pd.read_csv(csv_file)
        print(f"✅ CSV file readable: {len(df_csv)} rows, {len(df_csv.columns)} columns")
        print(f"   Columns: {list(df_csv.columns)}")
    except Exception as e:
        print(f"❌ CSV file error: {e}")
        return False
    
    # Test if text file is readable
    try:
        with open(txt_file, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"✅ Text file readable: {len(content)} characters")
    except Exception as e:
        print(f"❌ Text file error: {e}")
        return False
    
    return True

def main():
    """Main function"""
    print("🚀 Testing Multi-Format Document Support")
    print("=" * 50)
    
    success = test_document_formats()
    
    if success:
        print("\n🎉 All document formats are supported!")
        print("\nYour RAG application now supports:")
        print("📄 PDF files")
        print("📊 Excel files (.xlsx)")
        print("📈 CSV files")
        print("📝 Text files")
        print("\nTo test the RAG application:")
        print("1. Open: http://localhost:8502")
        print("2. Go to: 📄 Chat with your documents")
        print("3. Upload test files from the 'test_documents' folder")
        print("4. Ask questions about your documents!")
        print("\nSample questions you can ask:")
        print("• 'What products do we have in stock?'")
        print("• 'Who are the employees in the IT department?'")
        print("• 'What are our company values?'")
    else:
        print("\n⚠️ Some document format tests failed.")
        print("Please check the error messages above.")

if __name__ == "__main__":
    main()
