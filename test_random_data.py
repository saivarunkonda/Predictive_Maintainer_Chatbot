#!/usr/bin/env python3
"""
Test script to verify the random mixed data generation
"""
from streamlit_synthetic_generator import SyntheticDataGenerator

def test_random_generation():
    print("🧪 Testing Random Mixed Data Generation")
    print("=" * 50)
    
    generator = SyntheticDataGenerator()
    
    # Test connection
    is_connected, models = generator.check_connection()
    print(f"Ollama Connected: {'✅ Yes' if is_connected else '❌ No'}")
    
    if not is_connected:
        print("❌ Ollama not connected. Cannot test AI-generated content.")
        print("📊 Testing programmatic data generation instead...")
    
    # Test random mixed data generation
    try:
        print("\n🎲 Generating random mixed data (5 records)...")
        df = generator.generate_random_mixed_data(5)
        
        if not df.empty:
            print(f"✅ Successfully generated {len(df)} records")
            print(f"📋 Columns: {', '.join(df.columns.tolist())}")
            print(f"🎯 Entity types: {df['Entity_Type'].value_counts().to_dict()}")
            print("\n📊 Sample Data:")
            print(df.to_string(index=False))
            
            # Save to file for inspection
            df.to_csv("test_random_data.csv", index=False)
            print(f"\n💾 Data saved to: test_random_data.csv")
            
        else:
            print("❌ No data generated")
            
    except Exception as e:
        print(f"❌ Error generating random data: {e}")
    
    print("\n🎉 Test completed!")

if __name__ == "__main__":
    test_random_generation()
