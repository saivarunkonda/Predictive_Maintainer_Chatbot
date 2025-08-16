#!/usr/bin/env python3
"""
Streamlit Interface for Synthetic Data Generator
A web-based interface to generate synthetic data using Ollama
"""
import streamlit as st
import pandas as pd
import requests
import json
import time
import io
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import random
import string

# Configure Streamlit
st.set_page_config(
    page_title="Synthetic Data Generator",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Force light theme
st.markdown("""
<style>
    /* Force light theme colors */
    .stApp {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    /* Main content area */
    .main .block-container {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    /* Sidebar */
    .css-1d391kg {
        background-color: #f8f9fa !important;
        color: #000000 !important;
    }
    /* Text elements */
    p, span, div, label {
        color: #000000 !important;
    }
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #2c3e50 !important;
    }
</style>
""", unsafe_allow_html=True)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .data-card {
        background: #f8f9fa;
        color: #333333;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border: 1px solid #e9ecef;
    }
    .data-card h3 {
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .data-card p, .data-card li {
        color: #495057;
        line-height: 1.6;
    }
    .data-card ul {
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .metric-card h3 {
        color: white;
        font-size: 2rem;
        margin: 0;
    }
    .metric-card p {
        color: rgba(255, 255, 255, 0.9);
        margin: 0.5rem 0 0 0;
        font-size: 0.9rem;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #764ba2 0%, #667eea 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .stSelectbox > div > div {
        background-color: white;
        color: #333333;
    }
    .stTextInput > div > div > input {
        background-color: white;
        color: #333333;
    }
    .stTextArea > div > div > textarea {
        background-color: white;
        color: #333333;
    }
    .stSlider > div > div > div {
        color: #333333;
    }
    /* Fix sidebar text colors */
    .css-1d391kg {
        color: #333333;
    }
    .css-1v3fvcr {
        color: #333333;
    }
    /* Fix main content text colors */
    .stApp {
        background-color: #ffffff;
        color: #333333;
    }
    .css-1rs6os {
        color: #333333;
    }
    /* Fix dataframe text colors */
    .stDataFrame {
        color: #333333;
    }
    /* Fix metric display */
    .css-1wivap2 {
        color: #333333;
    }
    /* Fix success/error messages */
    .stAlert {
        color: #333333;
    }
    /* Fix subheader colors */
    .css-1v0mbdj {
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)

class SyntheticDataGenerator:
    def __init__(self, ollama_url: str = "http://localhost:11434", model: str = "llama3.2:3b"):
        self.ollama_url = ollama_url
        self.model = model
        self.timeout = 30
        
    def check_connection(self) -> tuple[bool, List[str]]:
        """Check if Ollama is accessible"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json()
                available_models = [m['name'] for m in models['models']]
                return True, available_models
            return False, []
        except Exception as e:
            st.error(f"Cannot connect to Ollama: {e}")
            return False, []
    
    def query_model(self, prompt: str, max_tokens: int = 100) -> Optional[str]:
        """Query the Ollama model with optimized settings"""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_ctx": 1024,
                        "num_gpu": 0,
                        "num_thread": 4,
                        "temperature": 0.7,
                        "top_k": 40,
                        "top_p": 0.9,
                        "repeat_penalty": 1.1,
                        "stop": ["\\n\\n", "---", "###"]
                    }
                },
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                st.error(f"Ollama returned status {response.status_code}")
                return None
                
        except requests.exceptions.Timeout:
            st.error("Request timed out. Model might be overloaded.")
            return None
        except Exception as e:
            st.error(f"Error querying model: {e}")
            return None
    
    def generate_employee_data(self, count: int) -> pd.DataFrame:
        """Generate employee data programmatically"""
        departments = ["IT", "Sales", "Marketing", "HR", "Finance", "Operations", "R&D"]
        positions = ["Manager", "Developer", "Analyst", "Specialist", "Director", "Coordinator"]
        first_names = ["John", "Jane", "Mike", "Sarah", "David", "Lisa", "Chris", "Emily", "Robert", "Anna"]
        last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez"]
        
        data = []
        for i in range(count):
            first_name = random.choice(first_names)
            last_name = random.choice(last_names)
            data.append({
                "Employee_ID": f"EMP{1000 + i}",
                "Name": f"{first_name} {last_name}",
                "Department": random.choice(departments),
                "Position": random.choice(positions),
                "Salary": random.randint(40000, 120000),
                "Years_Experience": random.randint(0, 20),
                "Performance_Rating": round(random.uniform(2.5, 5.0), 1),
                "Email": f"{first_name.lower()}.{last_name.lower()}@company.com",
                "Hire_Date": (datetime.now() - timedelta(days=random.randint(30, 2000))).strftime("%Y-%m-%d")
            })
        
        return pd.DataFrame(data)
    
    def generate_product_data(self, count: int) -> pd.DataFrame:
        """Generate product data programmatically"""
        categories = ["Electronics", "Clothing", "Home", "Books", "Sports", "Beauty", "Automotive"]
        brands = ["BrandA", "BrandB", "BrandC", "BrandD", "BrandE"]
        
        data = []
        for i in range(count):
            category = random.choice(categories)
            brand = random.choice(brands)
            data.append({
                "Product_ID": f"PRD{2000 + i}",
                "Name": f"{brand} {category} Item {i+1}",
                "Category": category,
                "Brand": brand,
                "Price": round(random.uniform(10.0, 500.0), 2),
                "Cost": round(random.uniform(5.0, 300.0), 2),
                "Stock": random.randint(0, 1000),
                "Rating": round(random.uniform(1.0, 5.0), 1),
                "Reviews": random.randint(0, 500),
                "Launch_Date": (datetime.now() - timedelta(days=random.randint(1, 365))).strftime("%Y-%m-%d")
            })
        
        return pd.DataFrame(data)
    
    def generate_customer_data(self, count: int) -> pd.DataFrame:
        """Generate customer data programmatically"""
        first_names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry", "Ivy", "Jack"]
        last_names = ["Anderson", "Baker", "Clark", "Davis", "Evans", "Foster", "Green", "Harris", "Irwin", "Jackson"]
        cities = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia", "San Antonio", "San Diego"]
        
        data = []
        for i in range(count):
            first_name = random.choice(first_names)
            last_name = random.choice(last_names)
            data.append({
                "Customer_ID": f"CUST{3000 + i}",
                "Name": f"{first_name} {last_name}",
                "Email": f"{first_name.lower()}.{last_name.lower()}@email.com",
                "Phone": f"+1-{random.randint(100, 999)}-{random.randint(100, 999)}-{random.randint(1000, 9999)}",
                "Age": random.randint(18, 80),
                "City": random.choice(cities),
                "Total_Orders": random.randint(0, 50),
                "Total_Spent": round(random.uniform(0, 5000), 2),
                "Registration_Date": (datetime.now() - timedelta(days=random.randint(1, 1000))).strftime("%Y-%m-%d"),
                "Status": random.choice(["Active", "Inactive", "Premium"])
            })
        
        return pd.DataFrame(data)
    
    def generate_transaction_data(self, count: int) -> pd.DataFrame:
        """Generate transaction data programmatically"""
        payment_methods = ["Credit Card", "Debit Card", "PayPal", "Bank Transfer", "Cash"]
        statuses = ["Completed", "Pending", "Failed", "Refunded"]
        
        data = []
        for i in range(count):
            data.append({
                "Transaction_ID": f"TXN{4000 + i}",
                "Customer_ID": f"CUST{random.randint(3000, 3100)}",
                "Product_ID": f"PRD{random.randint(2000, 2100)}",
                "Amount": round(random.uniform(10.0, 1000.0), 2),
                "Payment_Method": random.choice(payment_methods),
                "Status": random.choice(statuses),
                "Date": (datetime.now() - timedelta(days=random.randint(0, 365))).strftime("%Y-%m-%d"),
                "Time": f"{random.randint(0, 23):02d}:{random.randint(0, 59):02d}",
                "Discount": round(random.uniform(0, 20), 2),
                "Tax": round(random.uniform(5, 15), 2)
            })
        
        return pd.DataFrame(data)
    
    def generate_random_mixed_data(self, count: int) -> pd.DataFrame:
        """Generate random mixed data combining different types"""
        # Define different entity types with their generators
        entity_generators = [
            ("Person", self._generate_person_record),
            ("Product", self._generate_product_record),
            ("Transaction", self._generate_transaction_record),
            ("Event", self._generate_event_record),
            ("Review", self._generate_review_record)
        ]
        
        data = []
        for i in range(count):
            # Randomly select an entity type
            entity_type, generator_func = random.choice(entity_generators)
            record = generator_func(i)
            record["Entity_Type"] = entity_type
            record["Record_ID"] = f"RND{5000 + i}"
            data.append(record)
        
        return pd.DataFrame(data)
    
    def _generate_person_record(self, index: int) -> Dict[str, Any]:
        """Generate a person record"""
        first_names = ["Alex", "Jordan", "Taylor", "Casey", "Morgan", "Riley", "Avery", "Quinn"]
        last_names = ["Smith", "Johnson", "Brown", "Davis", "Wilson", "Moore", "Taylor", "Anderson"]
        
        first_name = random.choice(first_names)
        last_name = random.choice(last_names)
        age = random.randint(18, 70)
        
        return {
            "Name": f"{first_name} {last_name}",
            "Age": age,
            "Email": f"{first_name.lower()}.{last_name.lower()}{random.randint(1, 999)}@email.com",
            "Phone": f"+1-{random.randint(100, 999)}-{random.randint(100, 999)}-{random.randint(1000, 9999)}",
            "City": random.choice(["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"]),
            "Occupation": random.choice(["Engineer", "Teacher", "Doctor", "Artist", "Manager", "Analyst"]),
            "Salary": random.randint(30000, 150000) if age > 22 else None
        }
    
    def _generate_product_record(self, index: int) -> Dict[str, Any]:
        """Generate a product record"""
        categories = ["Electronics", "Books", "Clothing", "Home", "Sports", "Beauty"]
        adjectives = ["Premium", "Budget", "Professional", "Deluxe", "Standard", "Eco-friendly"]
        
        category = random.choice(categories)
        adjective = random.choice(adjectives)
        
        return {
            "Name": f"{adjective} {category} Item {index + 1}",
            "Category": category,
            "Price": round(random.uniform(5.99, 999.99), 2),
            "Rating": round(random.uniform(1.0, 5.0), 1),
            "In_Stock": random.choice([True, False]),
            "Stock_Count": random.randint(0, 500),
            "Manufacturer": f"Company{random.choice(['A', 'B', 'C', 'D', 'E'])}",
            "Weight": round(random.uniform(0.1, 50.0), 2)
        }
    
    def _generate_transaction_record(self, index: int) -> Dict[str, Any]:
        """Generate a transaction record"""
        return {
            "Amount": round(random.uniform(1.00, 5000.00), 2),
            "Currency": random.choice(["USD", "EUR", "GBP", "CAD", "AUD"]),
            "Payment_Method": random.choice(["Card", "PayPal", "Bank Transfer", "Cash", "Crypto"]),
            "Status": random.choice(["Success", "Pending", "Failed", "Refunded"]),
            "Merchant": f"Store{random.randint(1, 100)}",
            "Description": f"Purchase #{random.randint(10000, 99999)}",
            "Timestamp": (datetime.now() - timedelta(days=random.randint(0, 365))).strftime("%Y-%m-%d %H:%M:%S"),
            "Fee": round(random.uniform(0.00, 50.00), 2)
        }
    
    def _generate_event_record(self, index: int) -> Dict[str, Any]:
        """Generate an event record"""
        event_types = ["Conference", "Workshop", "Webinar", "Meeting", "Training", "Seminar"]
        statuses = ["Scheduled", "Completed", "Cancelled", "Postponed"]
        
        return {
            "Event_Name": f"{random.choice(event_types)} {index + 1}",
            "Type": random.choice(event_types),
            "Date": (datetime.now() + timedelta(days=random.randint(-30, 365))).strftime("%Y-%m-%d"),
            "Duration_Hours": random.randint(1, 8),
            "Attendees": random.randint(5, 500),
            "Location": random.choice(["Online", "New York", "San Francisco", "London", "Tokyo"]),
            "Status": random.choice(statuses),
            "Budget": random.randint(500, 50000)
        }
    
    def _generate_review_record(self, index: int) -> Dict[str, Any]:
        """Generate a review record"""
        sentiments = ["Positive", "Negative", "Neutral"]
        
        rating = random.randint(1, 5)
        sentiment = "Positive" if rating >= 4 else "Negative" if rating <= 2 else "Neutral"
        
        return {
            "Rating": rating,
            "Sentiment": sentiment,
            "Verified": random.choice([True, False]),
            "Helpful_Votes": random.randint(0, 100),
            "Review_Length": random.randint(10, 500),
            "Reviewer_Age": random.randint(18, 75),
            "Platform": random.choice(["Website", "App", "Third-party"]),
            "Date_Posted": (datetime.now() - timedelta(days=random.randint(1, 365))).strftime("%Y-%m-%d")
        }
    
    def generate_custom_data(self, description: str, count: int) -> pd.DataFrame:
        """Generate custom data based on user description using AI"""
        if not description.strip():
            st.error("Please provide a description for custom data generation")
            return pd.DataFrame()
        
        # Generate field names and types using AI
        fields_prompt = f"""Based on this description: "{description}"
        
Generate a JSON schema with field names and data types. Return only valid JSON format like this:
{{"field1": "string", "field2": "number", "field3": "date", "field4": "boolean"}}

Keep it simple with 4-8 fields maximum."""

        schema_response = self.query_model(fields_prompt, max_tokens=200)
        
        if not schema_response:
            st.error("Failed to generate schema from AI model")
            return self._fallback_custom_data(description, count)
        
        try:
            # Try to parse the JSON schema
            import json
            import re
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', schema_response, re.DOTALL)
            if json_match:
                schema_text = json_match.group()
                schema = json.loads(schema_text)
            else:
                raise ValueError("No JSON found in response")
            
            # Generate data based on schema
            data = []
            for i in range(count):
                record = {}
                for field_name, field_type in schema.items():
                    if field_type.lower() in ['string', 'text', 'name']:
                        # Generate text using AI
                        text_prompt = f"Generate a realistic {field_name} for {description}. One word or short phrase only:"
                        text_value = self.query_model(text_prompt, max_tokens=20)
                        record[field_name] = (text_value or f"Sample_{field_name}_{i}").strip()
                        
                    elif field_type.lower() in ['number', 'int', 'integer', 'float']:
                        record[field_name] = random.randint(1, 1000) if 'id' in field_name.lower() else round(random.uniform(1, 100), 2)
                        
                    elif field_type.lower() in ['date', 'datetime']:
                        record[field_name] = (datetime.now() - timedelta(days=random.randint(1, 365))).strftime("%Y-%m-%d")
                        
                    elif field_type.lower() in ['boolean', 'bool']:
                        record[field_name] = random.choice([True, False])
                        
                    else:
                        record[field_name] = f"Value_{i}"
                
                data.append(record)
                
                # Progress indicator
                if i % max(1, count // 10) == 0:
                    st.write(f"Generated {i+1}/{count} records...")
            
            return pd.DataFrame(data)
            
        except Exception as e:
            st.warning(f"Error parsing AI response: {e}. Using fallback method.")
            return self._fallback_custom_data(description, count)
    
    def _fallback_custom_data(self, description: str, count: int) -> pd.DataFrame:
        """Fallback method for custom data generation"""
        # Create basic schema based on common patterns
        data = []
        for i in range(count):
            record = {
                "ID": f"ITEM_{1000 + i}",
                "Name": f"Item {i+1}",
                "Description": description[:50] + f" #{i+1}",
                "Value": round(random.uniform(10, 1000), 2),
                "Category": random.choice(["Type A", "Type B", "Type C"]),
                "Date": (datetime.now() - timedelta(days=random.randint(1, 365))).strftime("%Y-%m-%d"),
                "Status": random.choice(["Active", "Inactive", "Pending"])
            }
            data.append(record)
        
        return pd.DataFrame(data)

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎲 Synthetic Data Generator</h1>
        <p>Generate realistic synthetic data using AI models</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize generator
    generator = SyntheticDataGenerator()
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Check connection
    is_connected, available_models = generator.check_connection()
    
    if is_connected:
        st.sidebar.success("✅ Connected to Ollama")
        st.sidebar.info(f"🤖 Model: {generator.model}")
        if available_models:
            st.sidebar.text(f"Available: {', '.join(available_models)}")
    else:
        st.sidebar.error("❌ Cannot connect to Ollama")
        st.sidebar.info("Please ensure Ollama is running")
        return
    
    # Data type selection
    st.sidebar.subheader("📊 Data Type")
    data_types = {
        "👥 Employees": "employees",
        "🛍️ Products": "products", 
        "🏠 Customers": "customers",
        "💳 Transactions": "transactions",
        "🎲 Random Mix": "random",
        "🎯 Custom": "custom"
    }
    
    selected_type = st.sidebar.selectbox(
        "Select data type to generate:",
        list(data_types.keys())
    )
    
    # Custom data type input
    custom_description = ""
    if selected_type == "🎯 Custom":
        st.sidebar.subheader("🔧 Custom Data Description")
        custom_description = st.sidebar.text_area(
            "Describe the data you want to generate:",
            placeholder="e.g., Student records with name, age, grade, subjects, GPA...",
            height=100
        )
        
        # Sample examples
        st.sidebar.markdown("**💡 Examples:**")
        st.sidebar.markdown("- Student records with grades")
        st.sidebar.markdown("- Restaurant menu items")
        st.sidebar.markdown("- Movie database")
        st.sidebar.markdown("- Vehicle inventory")
        st.sidebar.markdown("- Event bookings")
    
    # Number of records
    st.sidebar.subheader("📈 Settings")
    num_records = st.sidebar.slider(
        "Number of records:",
        min_value=1,
        max_value=1000,
        value=10,
        step=1
    )
    
    # Generate button
    generate_disabled = False
    if selected_type == "🎯 Custom" and not custom_description.strip():
        generate_disabled = True
        st.sidebar.warning("⚠️ Please provide a description for custom data")
    
    if st.sidebar.button("🚀 Generate Data", type="primary", disabled=generate_disabled):
        data_type = data_types[selected_type]
        
        with st.spinner(f"Generating {num_records} {data_type} records..."):
            start_time = time.time()
            
            try:
                # Generate data based on type
                if data_type == "employees":
                    df = generator.generate_employee_data(num_records)
                elif data_type == "products":
                    df = generator.generate_product_data(num_records)
                elif data_type == "customers":
                    df = generator.generate_customer_data(num_records)
                elif data_type == "transactions":
                    df = generator.generate_transaction_data(num_records)
                elif data_type == "random":
                    df = generator.generate_random_mixed_data(num_records)
                elif data_type == "custom":
                    df = generator.generate_custom_data(custom_description, num_records)
                else:
                    st.error(f"Unknown data type: {data_type}")
                    return
                
                generation_time = time.time() - start_time
                
                # Store in session state
                st.session_state.generated_data = df
                st.session_state.data_type = data_type
                st.session_state.generation_time = generation_time
                
                st.success(f"✅ Generated {len(df)} records in {generation_time:.2f} seconds!")
                
            except Exception as e:
                st.error(f"Error generating data: {e}")
    
    # Display generated data
    if hasattr(st.session_state, 'generated_data') and not st.session_state.generated_data.empty:
        df = st.session_state.generated_data
        data_type = st.session_state.data_type
        generation_time = st.session_state.generation_time
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{len(df)}</h3>
                <p>Records Generated</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{len(df.columns)}</h3>
                <p>Columns</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{generation_time:.2f}s</h3>
                <p>Generation Time</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{data_type.title()}</h3>
                <p>Data Type</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Data preview
        st.subheader("📋 Data Preview")
        st.dataframe(df, use_container_width=True)
        
        # Download options
        st.subheader("💾 Download Options")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # CSV download
            csv = df.to_csv(index=False)
            st.download_button(
                label="📄 Download CSV",
                data=csv,
                file_name=f"synthetic_{data_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Excel download
            buffer = io.BytesIO()
            df.to_excel(buffer, index=False)
            st.download_button(
                label="📊 Download Excel",
                data=buffer.getvalue(),
                file_name=f"synthetic_{data_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        with col3:
            # JSON download
            json_data = df.to_json(orient='records', indent=2)
            st.download_button(
                label="🔗 Download JSON",
                data=json_data,
                file_name=f"synthetic_{data_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        # Data statistics
        st.subheader("📊 Data Statistics")
        
        # Numeric columns statistics
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            st.write("**Numeric Columns:**")
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)
        
        # Categorical columns statistics
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            st.write("**Categorical Columns:**")
            for col in categorical_cols:
                if df[col].nunique() < 20:  # Only show if not too many unique values
                    st.write(f"**{col}:** {df[col].nunique()} unique values")
                    st.bar_chart(df[col].value_counts().head(10))
    
    else:
        # Welcome message
        st.markdown("""
        <div class="data-card">
            <h3>🚀 Welcome to Synthetic Data Generator!</h3>
            <p>Generate realistic synthetic data for testing, development, and analysis.</p>
            <h4>Available Data Types:</h4>
            <ul>
                <li><strong>👥 Employees:</strong> Employee records with names, departments, salaries, etc.</li>
                <li><strong>🛍️ Products:</strong> Product catalogs with prices, categories, ratings, etc.</li>
                <li><strong>🏠 Customers:</strong> Customer profiles with contact info, purchase history, etc.</li>
                <li><strong>💳 Transactions:</strong> Transaction records with payments, dates, amounts, etc.</li>
                <li><strong>🎲 Random Mix:</strong> Mixed dataset with random entity types (persons, products, transactions, events, reviews)</li>
                <li><strong>🎯 Custom:</strong> Generate any type of data based on your description!</li>
            </ul>
            <p>Select a data type from the sidebar and click "Generate Data" to get started!</p>
            <p><strong>💡 Try Custom:</strong> Describe any data you need like "student grades", "restaurant menu", "movie database", etc.</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
