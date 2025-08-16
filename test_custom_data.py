#!/usr/bin/env python3
"""
Test script for custom data generation
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from streamlit_synthetic_generator import SyntheticDataGenerator

def test_custom_generation():
    print("🧪 Testing Custom Data Generation")
    print("=" * 50)
    
    generator = SyntheticDataGenerator()
    
    # Test connection
    is_connected, models = generator.check_connection()
    if not is_connected:
        print("❌ Cannot connect to Ollama")
        return
    
    print("✅ Connected to Ollama")
    
    # Test custom data generation
    test_descriptions = [
        "Student records with name, age, grade, GPA, major",
        "Restaurant menu items with name, price, category, ingredients",
        "Movie database with title, director, year, genre, rating"
    ]
    
    for i, description in enumerate(test_descriptions, 1):
        print(f"\n{i}. Testing: {description}")
        print("-" * 40)
        
        try:
            df = generator.generate_custom_data(description, 3)
            
            if not df.empty:
                print(f"✅ Generated {len(df)} records")
                print(f"📊 Columns: {', '.join(df.columns.tolist())}")
                print("📝 Sample data:")
                print(df.head(1).to_string(index=False))
            else:
                print("❌ No data generated")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n🎉 Custom data generation test completed!")

if __name__ == "__main__":
    test_custom_generation()
