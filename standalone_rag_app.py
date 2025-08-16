#!/usr/bin/env python3
"""
Standalone RAG (Retrieval-Augmented Generation) Application
Supports PDF, Excel, CSV, and Text files with Ollama integration
"""
import os
import streamlit as st
import pandas as pd
import requests
import json
from io import StringIO
from pathlib import Path
import tempfile
from typing import List, Dict, Any
from datetime import datetime

# Configure Streamlit
st.set_page_config(
    page_title="Standalone RAG Application",
    page_icon="🤖",
    layout="wide"
)

# Add custom CSS for better styling
st.markdown("""
<style>
    .stApp {
        background-color: #f0f2f6;
    }
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .upload-section {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    .chat-container {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

class StandaloneRAGApp:
    def __init__(self):
        self.ollama_url = "http://localhost:11434"
        self.model_name = "llama3.2:3b"
        self.documents = []
        self.chat_history = []
        
        # Initialize session state
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'documents' not in st.session_state:
            st.session_state.documents = []
        if 'document_chunks' not in st.session_state:
            st.session_state.document_chunks = []
    
    def check_ollama_connection(self):
        """Check if Ollama is running and accessible"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json()
                available_models = [m['name'] for m in models['models']]
                return True, available_models
            return False, []
        except Exception as e:
            return False, []
    
    def load_pdf_content(self, file_path: str) -> List[str]:
        """Load PDF content (simplified version)"""
        try:
            # For demonstration, we'll return a placeholder
            # In a real implementation, you'd use PyPDF2 or similar
            return [f"PDF content from {file_path} would be extracted here"]
        except Exception as e:
            st.error(f"Error loading PDF: {e}")
            return []
    
    def load_excel_content(self, file_path: str) -> List[str]:
        """Load Excel file content"""
        try:
            df = pd.read_excel(file_path)
            content = []
            
            # Add column information
            content.append(f"Excel file contains {len(df)} rows and {len(df.columns)} columns.")
            content.append(f"Columns: {', '.join(df.columns.tolist())}")
            
            # Add data summary
            for idx, row in df.iterrows():
                row_text = f"Row {idx + 1}: " + ", ".join([f"{col}: {val}" for col, val in row.items()])
                content.append(row_text)
            
            # Add statistical summary if numeric columns exist
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                content.append("Statistical Summary:")
                for col in numeric_cols:
                    content.append(f"{col}: min={df[col].min()}, max={df[col].max()}, mean={df[col].mean():.2f}")
            
            return content
        except Exception as e:
            st.error(f"Error loading Excel file: {e}")
            return []
    
    def load_csv_content(self, file_path: str) -> List[str]:
        """Load CSV file content"""
        try:
            df = pd.read_csv(file_path)
            content = []
            
            # Add column information
            content.append(f"CSV file contains {len(df)} rows and {len(df.columns)} columns.")
            content.append(f"Columns: {', '.join(df.columns.tolist())}")
            
            # Add data summary
            for idx, row in df.iterrows():
                row_text = f"Row {idx + 1}: " + ", ".join([f"{col}: {val}" for col, val in row.items()])
                content.append(row_text)
            
            return content
        except Exception as e:
            st.error(f"Error loading CSV file: {e}")
            return []
    
    def load_text_content(self, file_path: str) -> List[str]:
        """Load text file content"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split into chunks (simple paragraph-based splitting)
            chunks = [chunk.strip() for chunk in content.split('\n\n') if chunk.strip()]
            return chunks
        except Exception as e:
            st.error(f"Error loading text file: {e}")
            return []
    
    def process_uploaded_file(self, uploaded_file) -> List[str]:
        """Process uploaded file and extract content"""
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            if file_extension == 'pdf':
                content = self.load_pdf_content(tmp_file_path)
            elif file_extension in ['xlsx', 'xls']:
                content = self.load_excel_content(tmp_file_path)
            elif file_extension == 'csv':
                content = self.load_csv_content(tmp_file_path)
            elif file_extension == 'txt':
                content = self.load_text_content(tmp_file_path)
            else:
                st.error(f"Unsupported file format: {file_extension}")
                return []
            
            # Clean up temporary file
            os.unlink(tmp_file_path)
            return content
            
        except Exception as e:
            st.error(f"Error processing file: {e}")
            # Clean up temporary file
            try:
                os.unlink(tmp_file_path)
            except:
                pass
            return []
    
    def find_relevant_chunks(self, query: str, chunks: List[str], max_chunks: int = 3) -> List[str]:
        """Simple keyword-based chunk retrieval"""
        query_lower = query.lower()
        query_words = query_lower.split()
        
        # Score chunks based on keyword matches
        chunk_scores = []
        for chunk in chunks:
            chunk_lower = chunk.lower()
            score = sum(1 for word in query_words if word in chunk_lower)
            chunk_scores.append((chunk, score))
        
        # Sort by score and return top chunks
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        relevant_chunks = [chunk for chunk, score in chunk_scores[:max_chunks] if score > 0]
        
        return relevant_chunks
    
    def query_ollama(self, prompt: str, context: str = "") -> str:
        """Query Ollama with context"""
        try:
            full_prompt = f"""Context: {context}

Question: {prompt}

Please answer the question based on the provided context. If the answer is not in the context, say so clearly."""
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "num_ctx": 2048,
                        "num_gpu": 0,
                        "num_thread": 4,
                        "temperature": 0.1
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', 'No response received')
            else:
                return f"Error: Ollama returned status {response.status_code}"
                
        except Exception as e:
            return f"Error querying Ollama: {e}"
    
    def render_header(self):
        """Render the application header"""
        st.markdown("""
        <div class="main-header">
            <h1>🤖 Standalone RAG Application</h1>
            <p>Upload documents (PDF, Excel, CSV, Text) and ask questions about them!</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render the sidebar with upload functionality"""
        st.sidebar.header("📁 Document Upload")
        
        # Check Ollama connection
        is_connected, available_models = self.check_ollama_connection()
        
        if is_connected:
            st.sidebar.success(f"✅ Connected to Ollama")
            st.sidebar.info(f"🤖 Model: {self.model_name}")
            if self.model_name not in available_models:
                st.sidebar.warning(f"⚠️ Model {self.model_name} not found in available models: {available_models}")
        else:
            st.sidebar.error("❌ Cannot connect to Ollama")
            st.sidebar.info("Please ensure Ollama is running: `ollama serve`")
        
        # File upload
        uploaded_files = st.sidebar.file_uploader(
            "Upload Documents",
            type=['pdf', 'xlsx', 'xls', 'csv', 'txt'],
            accept_multiple_files=True,
            help="Supported formats: PDF, Excel, CSV, Text"
        )
        
        if uploaded_files:
            st.sidebar.success(f"📁 {len(uploaded_files)} file(s) uploaded")
            
            if st.sidebar.button("Process Documents"):
                self.process_documents(uploaded_files)
        
        # Display processed documents
        if st.session_state.documents:
            st.sidebar.subheader("📋 Processed Documents")
            for doc in st.session_state.documents:
                st.sidebar.text(f"• {doc['name']} ({len(doc['chunks'])} chunks)")
    
    def process_documents(self, uploaded_files):
        """Process uploaded documents"""
        st.session_state.documents = []
        st.session_state.document_chunks = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing {uploaded_file.name}...")
            
            chunks = self.process_uploaded_file(uploaded_file)
            
            if chunks:
                st.session_state.documents.append({
                    'name': uploaded_file.name,
                    'chunks': chunks,
                    'processed_at': datetime.now()
                })
                st.session_state.document_chunks.extend(chunks)
                st.success(f"✅ Processed {uploaded_file.name} ({len(chunks)} chunks)")
            else:
                st.error(f"❌ Failed to process {uploaded_file.name}")
            
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        status_text.text("Processing complete!")
        st.rerun()
    
    def render_chat_interface(self):
        """Render the chat interface"""
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            if not st.session_state.document_chunks:
                st.error("Please upload and process some documents first!")
                return
            
            # Add user message to chat
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.write(prompt)
            
            # Find relevant chunks
            relevant_chunks = self.find_relevant_chunks(prompt, st.session_state.document_chunks)
            
            if not relevant_chunks:
                response = "I couldn't find relevant information in the uploaded documents to answer your question."
            else:
                # Create context from relevant chunks
                context = "\n\n".join(relevant_chunks)
                
                # Query Ollama
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = self.query_ollama(prompt, context)
                    st.write(response)
                    
                    # Show sources
                    with st.expander("📚 Sources"):
                        for i, chunk in enumerate(relevant_chunks, 1):
                            st.text(f"Source {i}:")
                            st.text(chunk[:200] + "..." if len(chunk) > 200 else chunk)
                            st.divider()
            
            # Add assistant response to chat
            st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def render_sample_questions(self):
        """Render sample questions section"""
        if st.session_state.documents:
            st.subheader("💡 Sample Questions")
            
            sample_questions = [
                "What information is contained in the documents?",
                "Can you summarize the key points?",
                "What are the main categories or topics?",
                "Are there any numerical data or statistics?",
                "What products or items are mentioned?"
            ]
            
            cols = st.columns(len(sample_questions))
            for i, question in enumerate(sample_questions):
                with cols[i]:
                    if st.button(question, key=f"sample_{i}"):
                        # Simulate clicking the question
                        st.session_state.sample_question = question
                        st.rerun()
    
    def run(self):
        """Main application runner"""
        self.render_header()
        self.render_sidebar()
        
        if not st.session_state.documents:
            st.markdown("""
            <div class="upload-section">
                <h3>🚀 Getting Started</h3>
                <p>1. Upload your documents using the sidebar</p>
                <p>2. Click "Process Documents" to analyze them</p>
                <p>3. Ask questions about your documents in the chat below</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            self.render_sample_questions()
        
        self.render_chat_interface()
        
        # Handle sample question selection
        if hasattr(st.session_state, 'sample_question'):
            prompt = st.session_state.sample_question
            delattr(st.session_state, 'sample_question')
            
            if st.session_state.document_chunks:
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                
                relevant_chunks = self.find_relevant_chunks(prompt, st.session_state.document_chunks)
                context = "\n\n".join(relevant_chunks)
                response = self.query_ollama(prompt, context)
                
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.rerun()

def main():
    """Main function"""
    app = StandaloneRAGApp()
    app.run()

if __name__ == "__main__":
    main()
