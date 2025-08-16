#!/usr/bin/env python3
"""
Standalone Synthetic Data Generator using Ollama
Generates various types of synthetic data using the llama3.2:3b model
"""
import requests
import json
import pandas as pd
import random
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import sys

class SyntheticDataGenerator:
    def __init__(self, ollama_url: str = "http://localhost:11434", model: str = "llama3.2:3b"):
        self.ollama_url = ollama_url
        self.model = model
        self.timeout = 30  # Reduced timeout
        
    def check_connection(self) -> bool:
        """Check if Ollama is accessible"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json()
                available_models = [m['name'] for m in models['models']]
                if self.model in available_models:
                    print(f"✅ Connected to Ollama. Model {self.model} is available.")
                    return True
                else:
                    print(f"❌ Model {self.model} not found. Available: {available_models}")
                    return False
            return False
        except Exception as e:
            print(f"❌ Cannot connect to Ollama: {e}")
            return False
    
    def query_model(self, prompt: str, max_tokens: int = 100) -> Optional[str]:
        """Query the Ollama model with optimized settings"""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_ctx": 1024,      # Smaller context
                    "num_gpu": 0,         # Force CPU
                    "num_thread": 2,      # Limit threads
                    "temperature": 0.7,   # Add some creativity
                    "top_p": 0.9,
                    "top_k": 40,
                    "num_predict": max_tokens  # Limit output length
                }
            }
            
            print(f"🤖 Generating response... (timeout: {self.timeout}s)")
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                print(f"❌ API Error: Status {response.status_code}")
                return None
                
        except requests.exceptions.Timeout:
            print("⏰ Request timed out. The model might be busy or the prompt too complex.")
            return None
        except Exception as e:
            print(f"❌ Error querying model: {e}")
            return None
    
    def generate_simple_names(self, count: int = 5) -> List[str]:
        """Generate simple names using basic prompts"""
        prompt = f"Generate {count} realistic first names, one per line. Only names, no numbers or explanations:"
        
        response = self.query_model(prompt, max_tokens=50)
        if response:
            names = [name.strip() for name in response.split('\n') if name.strip()]
            return names[:count]
        
        # Fallback to predefined names if model fails
        fallback_names = ["John", "Jane", "Michael", "Sarah", "David", "Emma", "Chris", "Lisa", "Tom", "Anna"]
        return random.sample(fallback_names, min(count, len(fallback_names)))
    
    def generate_product_data(self, count: int = 5) -> pd.DataFrame:
        """Generate product data"""
        print(f"📦 Generating {count} product records...")
        
        # Simple, focused prompts
        products = []
        
        product_types = ["Laptop", "Mouse", "Keyboard", "Monitor", "Headphones", "Webcam", "Speaker", "Tablet"]
        
        for i in range(count):
            product_type = random.choice(product_types)
            
            # Generate price using simple prompt
            price_prompt = f"What would be a realistic price in USD for a {product_type}? Give only the number without $ or explanations:"
            price_response = self.query_model(price_prompt, max_tokens=20)
            
            try:
                # Extract price from response
                price = float(''.join(filter(lambda x: x.isdigit() or x == '.', price_response or "99.99")))
                if price == 0:
                    price = random.uniform(50, 500)
            except:
                price = random.uniform(50, 500)
            
            # Generate description using simple prompt
            desc_prompt = f"Write a short product description for a {product_type} in one sentence:"
            description = self.query_model(desc_prompt, max_tokens=30) or f"High-quality {product_type}"
            
            products.append({
                'Product': product_type,
                'Price': round(price, 2),
                'Category': 'Electronics',
                'Stock': random.randint(10, 100),
                'Rating': round(random.uniform(3.5, 5.0), 1),
                'Description': description
            })
            
            print(f"  ✅ Generated product {i+1}/{count}")
            time.sleep(1)  # Small delay to prevent overwhelming the model
        
        return pd.DataFrame(products)
    
    def generate_employee_data(self, count: int = 5) -> pd.DataFrame:
        """Generate employee data"""
        print(f"👥 Generating {count} employee records...")
        
        employees = []
        departments = ["IT", "Sales", "Marketing", "HR", "Finance"]
        
        names = self.generate_simple_names(count)
        
        for i, name in enumerate(names):
            # Simple random generation for other fields
            employees.append({
                'Name': name,
                'Department': random.choice(departments),
                'Salary': random.randint(40000, 120000),
                'Years_Experience': random.randint(1, 15),
                'Performance': random.choice(['Excellent', 'Good', 'Average'])
            })
            
            print(f"  ✅ Generated employee {i+1}/{count}")
        
        return pd.DataFrame(employees)
    
    def generate_customer_reviews(self, count: int = 3) -> pd.DataFrame:
        """Generate customer reviews"""
        print(f"⭐ Generating {count} customer reviews...")
        
        reviews = []
        products = ["Laptop", "Mouse", "Keyboard"]
        
        for i in range(count):
            product = random.choice(products)
            rating = random.randint(3, 5)
            
            # Simple review prompt
            review_prompt = f"Write a brief {rating}-star customer review for a {product} in one sentence:"
            review_text = self.query_model(review_prompt, max_tokens=40)
            
            if not review_text:
                review_text = f"Great {product}, would recommend to others."
            
            reviews.append({
                'Product': product,
                'Rating': rating,
                'Review': review_text,
                'Date': (datetime.now() - timedelta(days=random.randint(1, 90))).strftime('%Y-%m-%d')
            })
            
            print(f"  ✅ Generated review {i+1}/{count}")
            time.sleep(1)
        
        return pd.DataFrame(reviews)
    
    def generate_sales_data(self, count: int = 10) -> pd.DataFrame:
        """Generate sales transaction data"""
        print(f"💰 Generating {count} sales records...")
        
        sales = []
        products = ["Laptop", "Mouse", "Keyboard", "Monitor", "Headphones"]
        
        for i in range(count):
            product = random.choice(products)
            quantity = random.randint(1, 5)
            unit_price = random.uniform(25, 1000)
            
            sales.append({
                'Transaction_ID': f"T{1000 + i}",
                'Product': product,
                'Quantity': quantity,
                'Unit_Price': round(unit_price, 2),
                'Total': round(quantity * unit_price, 2),
                'Date': (datetime.now() - timedelta(days=random.randint(1, 30))).strftime('%Y-%m-%d'),
                'Customer_ID': f"C{random.randint(1001, 9999)}"
            })
            
            print(f"  ✅ Generated sale {i+1}/{count}")
        
        return pd.DataFrame(sales)
    
    def save_to_files(self, dataframes: Dict[str, pd.DataFrame], output_dir: str = "synthetic_data"):
        """Save generated data to files"""
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        for name, df in dataframes.items():
            # Save as Excel
            excel_path = f"{output_dir}/{name}.xlsx"
            df.to_excel(excel_path, index=False)
            print(f"✅ Saved {excel_path}")
            
            # Save as CSV
            csv_path = f"{output_dir}/{name}.csv"
            df.to_csv(csv_path, index=False)
            print(f"✅ Saved {csv_path}")
    
    def generate_all_data(self):
        """Generate all types of synthetic data"""
        print("🚀 Starting Synthetic Data Generation")
        print("=" * 50)
        
        if not self.check_connection():
            print("❌ Cannot connect to Ollama. Please ensure it's running.")
            return
        
        dataframes = {}
        
        try:
            # Generate different types of data
            dataframes['products'] = self.generate_product_data(3)  # Reduced count
            dataframes['employees'] = self.generate_employee_data(3)
            dataframes['reviews'] = self.generate_customer_reviews(2)
            dataframes['sales'] = self.generate_sales_data(5)
            
            # Save all data
            self.save_to_files(dataframes)
            
            print("\n" + "=" * 50)
            print("🎉 Synthetic Data Generation Complete!")
            print("\nGenerated datasets:")
            for name, df in dataframes.items():
                print(f"  📊 {name}: {len(df)} records")
            
            print(f"\n📁 Files saved in 'synthetic_data' directory")
            print("📊 Available formats: Excel (.xlsx) and CSV (.csv)")
            
        except Exception as e:
            print(f"❌ Error during data generation: {e}")

def main():
    """Main function with interactive menu"""
    generator = SyntheticDataGenerator()
    
    while True:
        print("\n" + "="*50)
        print("🤖 Synthetic Data Generator")
        print("="*50)
        print("1. Check Ollama Connection")
        print("2. Generate Product Data")
        print("3. Generate Employee Data") 
        print("4. Generate Customer Reviews")
        print("5. Generate Sales Data")
        print("6. Generate All Data Types")
        print("7. Exit")
        print("="*50)
        
        choice = input("Select an option (1-7): ").strip()
        
        if choice == '1':
            generator.check_connection()
            
        elif choice == '2':
            count = int(input("How many products to generate? (default 3): ") or 3)
            df = generator.generate_product_data(count)
            print(f"\n📊 Generated Products:")
            print(df.to_string(index=False))
            
        elif choice == '3':
            count = int(input("How many employees to generate? (default 3): ") or 3)
            df = generator.generate_employee_data(count)
            print(f"\n📊 Generated Employees:")
            print(df.to_string(index=False))
            
        elif choice == '4':
            count = int(input("How many reviews to generate? (default 2): ") or 2)
            df = generator.generate_customer_reviews(count)
            print(f"\n📊 Generated Reviews:")
            print(df.to_string(index=False))
            
        elif choice == '5':
            count = int(input("How many sales records to generate? (default 5): ") or 5)
            df = generator.generate_sales_data(count)
            print(f"\n📊 Generated Sales:")
            print(df.to_string(index=False))
            
        elif choice == '6':
            generator.generate_all_data()
            
        elif choice == '7':
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please select 1-7.")

if __name__ == "__main__":
    main()
