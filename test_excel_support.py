#!/usr/bin/env python3
"""
Test script to verify Excel and CSV support in the RAG application
"""
import pandas as pd
import os

def create_test_files():
    """Create test Excel and CSV files"""
    
    # Create test data
    data = {
        'Product': ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Speakers'],
        'Price': [999.99, 25.50, 75.00, 299.99, 150.00],
        'Category': ['Electronics', 'Accessories', 'Accessories', 'Electronics', 'Electronics'],
        'Stock': [50, 200, 100, 75, 30],
        'Rating': [4.5, 4.2, 4.7, 4.3, 4.1]
    }
    
    df = pd.DataFrame(data)
    
    # Create tmp directory if it doesn't exist
    if not os.path.exists('tmp'):
        os.makedirs('tmp')
    
    # Save as Excel file
    excel_path = 'tmp/test_products.xlsx'
    df.to_excel(excel_path, index=False, sheet_name='Products')
    print(f"✅ Created test Excel file: {excel_path}")
    
    # Save as CSV file
    csv_path = 'tmp/test_products.csv'
    df.to_csv(csv_path, index=False)
    print(f"✅ Created test CSV file: {csv_path}")
    
    # Create a text file
    txt_path = 'tmp/test_document.txt'
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("""Product Information Document

This document contains information about our product inventory.

Our Electronics Category includes:
- Laptops: High-performance computing devices
- Monitors: Display screens for computers
- Speakers: Audio output devices

Our Accessories Category includes:
- Mouse: Input devices for computers
- Keyboard: Input devices for typing

All products are available with competitive pricing and excellent customer ratings.
Contact our sales team for bulk orders and special discounts.
""")
    print(f"✅ Created test text file: {txt_path}")
    
    return excel_path, csv_path, txt_path

def test_document_loading():
    """Test the document loading functionality"""
    try:
        # Import the document loader from your application
        import sys
        sys.path.append('pages')
        
        # We'll test this when the Streamlit app is running
        print("📝 Test files created successfully!")
        print("🚀 Now you can test the RAG application with these files:")
        print("   1. Run: streamlit run Home.py")
        print("   2. Navigate to: 📄 Chat with your documents")
        print("   3. Upload the test files from the 'tmp' folder")
        print("   4. Try asking questions like:")
        print("      - 'What products do you have?'")
        print("      - 'What is the price of the laptop?'")
        print("      - 'Which products are in the Electronics category?'")
        print("      - 'What are the ratings for each product?'")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🧪 Creating test files for enhanced RAG application...")
    print("=" * 60)
    
    excel_path, csv_path, txt_path = create_test_files()
    test_document_loading()
    
    print("\n" + "=" * 60)
    print("✅ Test setup complete!")
