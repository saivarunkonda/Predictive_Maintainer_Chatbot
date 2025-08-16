#!/usr/bin/env python3
"""
Test script to verify Ollama embeddings work for document chat
"""
import os
import sys
import streamlit as st
from langchain_community.embeddings import OllamaEmbeddings

def test_ollama_embeddings():
    """Test Ollama embeddings directly"""
    print("🧪 Testing Ollama Embeddings for Document Chat")
    print("=" * 60)
    
    try:
        # Test Ollama connection
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama is running")
            models = response.json()
            available_models = [m['name'] for m in models['models']]
            print(f"📋 Available models: {available_models}")
            
            if "llama3.2:3b" in available_models:
                print("✅ llama3.2:3b model is available")
            else:
                print("❌ llama3.2:3b model not found")
                return
        else:
            print("❌ Ollama is not running")
            return
            
    except Exception as e:
        print(f"❌ Error connecting to Ollama: {e}")
        return
    
    try:
        # Test Ollama embeddings
        print("\n🔗 Testing Ollama embeddings...")
        embeddings = OllamaEmbeddings(
            base_url="http://localhost:11434",
            model="llama3.2:3b"
        )
        
        # Test with a simple query
        test_text = "This is a test document for embeddings."
        print(f"📝 Test text: '{test_text}'")
        
        embedding_vector = embeddings.embed_query(test_text)
        print(f"✅ Embedding generated successfully!")
        print(f"📏 Embedding dimension: {len(embedding_vector)}")
        print(f"🔢 First 5 values: {embedding_vector[:5]}")
        
        # Test with multiple documents
        test_docs = [
            "This is the first document.",
            "This is the second document with different content.",
            "A third document about completely different topics."
        ]
        
        print(f"\n📚 Testing batch embeddings with {len(test_docs)} documents...")
        doc_embeddings = embeddings.embed_documents(test_docs)
        print(f"✅ Batch embeddings generated successfully!")
        print(f"📊 Generated {len(doc_embeddings)} embeddings")
        
        return True
        
    except Exception as e:
        print(f"❌ Error with Ollama embeddings: {e}")
        return False

def test_document_workflow():
    """Test the complete document workflow"""
    print("\n🔄 Testing Document Processing Workflow")
    print("=" * 50)
    
    try:
        # Create a sample document
        sample_content = """
        This is a sample document for testing the RAG system.
        It contains information about artificial intelligence and machine learning.
        
        Artificial Intelligence (AI) is a broad field of computer science concerned with 
        building smart machines capable of performing tasks that typically require human intelligence.
        
        Machine Learning is a subset of AI that enables machines to learn and improve from 
        experience without being explicitly programmed.
        """
        
        # Save sample document
        os.makedirs("test_docs", exist_ok=True)
        sample_file = "test_docs/sample_document.txt"
        with open(sample_file, "w", encoding="utf-8") as f:
            f.write(sample_content)
        
        print(f"📄 Created sample document: {sample_file}")
        
        # Test document loading
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(sample_file, encoding='utf-8')
        docs = loader.load()
        print(f"✅ Document loaded: {len(docs)} pages")
        
        # Test text splitting
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=50
        )
        splits = text_splitter.split_documents(docs)
        print(f"✅ Document split into {len(splits)} chunks")
        
        # Test vector store creation
        from langchain_community.vectorstores import DocArrayInMemorySearch
        
        embeddings = OllamaEmbeddings(
            base_url="http://localhost:11434",
            model="llama3.2:3b"
        )
        
        print("🔗 Creating vector store...")
        vectordb = DocArrayInMemorySearch.from_documents(splits, embeddings)
        print("✅ Vector store created successfully!")
        
        # Test retrieval
        query = "What is artificial intelligence?"
        print(f"\n🔍 Testing retrieval with query: '{query}'")
        
        retriever = vectordb.as_retriever(search_kwargs={'k': 2})
        retrieved_docs = retriever.get_relevant_documents(query)
        
        print(f"✅ Retrieved {len(retrieved_docs)} relevant documents")
        for i, doc in enumerate(retrieved_docs, 1):
            print(f"📋 Document {i}: {doc.page_content[:100]}...")
        
        # Cleanup
        if os.path.exists(sample_file):
            os.remove(sample_file)
        if os.path.exists("test_docs"):
            os.rmdir("test_docs")
        
        print("\n🎉 Document workflow test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Document workflow test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Ollama Document Chat Tests\n")
    
    # Test 1: Basic Ollama embeddings
    embedding_test = test_ollama_embeddings()
    
    if embedding_test:
        # Test 2: Full document workflow
        workflow_test = test_document_workflow()
        
        if workflow_test:
            print("\n🎊 ALL TESTS PASSED!")
            print("✅ Your document chat should work with Ollama embeddings")
            print("💡 Run the Streamlit app: streamlit run pages/4_📄_chat_with_your_documents.py")
        else:
            print("\n⚠️ Embedding test passed but workflow failed")
    else:
        print("\n❌ Embedding test failed - check Ollama setup")
        print("💡 Make sure Ollama is running: ollama serve")
        print("💡 Make sure llama3.2:3b model is installed: ollama pull llama3.2:3b")
