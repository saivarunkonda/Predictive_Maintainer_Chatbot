#!/usr/bin/env python3
"""
Test script for the synthetic data generator
"""
from synthetic_data_generator import SyntheticDataGenerator
import time

def test_data_generation():
    """Test the data generation functionality"""
    print("🧪 Testing Synthetic Data Generator")
    print("=" * 50)
    
    generator = SyntheticDataGenerator()
    
    # Test connection
    is_connected = generator.check_connection()
    print(f"Ollama Connected: {'✅ Yes' if is_connected else '❌ No'}")
    
    # Test each data type
    data_types = {
        "employees": "generate_employee_data",
        "products": "generate_product_data", 
        "customers": "generate_customer_reviews",
        "transactions": "generate_sales_data"
    }
    
    for data_type, method_name in data_types.items():
        print(f"\n🔄 Testing {data_type} generation...")
        start_time = time.time()
        
        try:
            # Generate small sample
            method = getattr(generator, method_name)
            df = method(5)
            generation_time = time.time() - start_time
            
            if not df.empty:
                print(f"✅ Generated {len(df)} {data_type} records in {generation_time:.2f}s")
                print(f"   Columns: {', '.join(df.columns.tolist())}")
                print(f"   Sample: {df.iloc[0].to_dict()}")
            else:
                print(f"❌ Failed to generate {data_type} data")
                
        except Exception as e:
            print(f"❌ Error generating {data_type}: {e}")
    
    print(f"\n🎉 Test completed!")

if __name__ == "__main__":
    test_data_generation()
