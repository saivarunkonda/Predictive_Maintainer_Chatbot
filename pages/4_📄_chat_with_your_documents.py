import os
import utils
import streamlit as st
from streaming import StreamHandler
import pandas as pd
from io import StringIO

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import DocArrayInMemorySearch, Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document


st.set_page_config(page_title="ChatDocuments", page_icon="📄")
st.header('Chat with your documents (Enhanced RAG)')
st.write('Has access to custom documents (PDF, Excel, CSV, TXT) and can respond to user queries by referring to the content within those documents')
st.write('[![view source code ](https://img.shields.io/badge/view_source_code-gray?logo=github)](https://github.com/shashankdeshpande/langchain-chatbot/blob/master/pages/4_%F0%9F%93%84_chat_with_your_documents.py)')

class CustomDocChatbot:

    def __init__(self):
        utils.sync_st_session()
        self.llm = utils.configure_llm()
        self.embedding_model = utils.configure_embedding_model()

    def load_document(self, file_path, file_name):
        """Load document based on file extension"""
        file_extension = os.path.splitext(file_name)[1].lower()
        
        try:
            if file_extension == '.pdf':
                loader = PyPDFLoader(file_path)
                return loader.load()
            
            elif file_extension == '.txt':
                loader = TextLoader(file_path, encoding='utf-8')
                return loader.load()
            
            elif file_extension in ['.xlsx', '.xls']:
                # Handle Excel files
                excel_data = pd.read_excel(file_path, sheet_name=None)  # Read all sheets
                docs = []
                
                for sheet_name, df in excel_data.items():
                    # Convert DataFrame to text
                    text_content = f"Sheet: {sheet_name}\n\n"
                    
                    # Add column headers
                    text_content += "Columns: " + ", ".join(df.columns.astype(str)) + "\n\n"
                    
                    # Add data rows
                    for idx, row in df.iterrows():
                        row_text = " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                        text_content += f"Row {idx + 1}: {row_text}\n"
                    
                    # Create Document object
                    doc = Document(
                        page_content=text_content,
                        metadata={"source": file_path, "sheet": sheet_name, "file_type": "excel"}
                    )
                    docs.append(doc)
                
                return docs
            
            elif file_extension == '.csv':
                # Handle CSV files
                df = pd.read_csv(file_path)
                
                # Convert DataFrame to text
                text_content = "CSV Data:\n\n"
                text_content += "Columns: " + ", ".join(df.columns.astype(str)) + "\n\n"
                
                for idx, row in df.iterrows():
                    row_text = " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                    text_content += f"Row {idx + 1}: {row_text}\n"
                
                # Create Document object
                doc = Document(
                    page_content=text_content,
                    metadata={"source": file_path, "file_type": "csv"}
                )
                return [doc]
            
            else:
                st.error(f"Unsupported file format: {file_extension}")
                return []
                
        except Exception as e:
            st.error(f"Error loading {file_name}: {str(e)}")
            return []

    def save_file(self, file):
        folder = 'tmp'
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        file_path = f'./{folder}/{file.name}'
        with open(file_path, 'wb') as f:
            f.write(file.getvalue())
        return file_path

    @st.spinner('Analyzing documents..')
    def setup_qa_chain(self, uploaded_files):
        # Load documents
        docs = []
        for file in uploaded_files:
            file_path = self.save_file(file)
            file_docs = self.load_document(file_path, file.name)
            docs.extend(file_docs)
        
        if not docs:
            st.error("No documents could be loaded. Please check your files.")
            st.stop()
        
        # Split documents and store in vector db (optimized for Ollama)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Optimal size for Ollama models
            chunk_overlap=200,  # Better overlap for context
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        splits = text_splitter.split_documents(docs)
        
        # Show document processing info
        st.info(f"📄 Processing {len(docs)} documents into {len(splits)} chunks")
        
        # Limit the number of chunks to prevent memory issues
        max_chunks = 200  # Increased for better coverage
        if len(splits) > max_chunks:
            st.warning(f"⚠️ Document has {len(splits)} chunks. Limiting to {max_chunks} chunks to prevent memory issues.")
            splits = splits[:max_chunks]
        
        # Process in batches to avoid memory issues
        try:
            vectordb = DocArrayInMemorySearch.from_documents(splits, self.embedding_model)
        except Exception as e:
            st.error(f"❌ Memory error with DocArray: {str(e)}")
            st.info("🔄 Trying alternative vector store (Chroma)...")
            try:
                # Alternative: Use Chroma which is more memory efficient
                vectordb = Chroma.from_documents(
                    documents=splits,
                    embedding=self.embedding_model,
                    persist_directory=None  # In-memory
                )
            except Exception as e2:
                st.error(f"❌ All vector stores failed: {str(e2)}")
                st.error("Solutions:")
                st.error("1. Upload smaller documents")
                st.error("2. Restart the application")
                st.error("3. Ensure you have sufficient RAM")
                st.stop()

        # Define retriever with better settings for Ollama
        retriever = vectordb.as_retriever(
            search_type='mmr',  # Maximum Marginal Relevance for diverse results
            search_kwargs={'k': 3, 'fetch_k': 6}  # Retrieve more context for better responses
        )

        # Setup memory for contextual conversation        
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            output_key='answer',
            return_messages=True
        )

        # Setup LLM and QA chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            verbose=False
        )
        return qa_chain

    @utils.enable_chat_history
    def main(self):
        # Add model selection info
        st.sidebar.subheader("🤖 Model Configuration")
        if hasattr(self.llm, 'model'):
            st.sidebar.info(f"**Model**: {self.llm.model}")
        elif hasattr(self.llm, 'model_name'):
            st.sidebar.info(f"**Model**: {self.llm.model_name}")
        
        # Add RAG settings
        st.sidebar.subheader("📚 RAG Settings")
        chunk_size = st.sidebar.slider("Chunk Size", 500, 2000, 1000, 100)
        max_chunks = st.sidebar.slider("Max Chunks", 50, 500, 200, 50)
        
        # User Inputs
        uploaded_files = st.sidebar.file_uploader(
            label='Upload Documents', 
            type=['pdf', 'txt', 'xlsx', 'xls', 'csv'], 
            accept_multiple_files=True,
            help="Supported formats: PDF, TXT, Excel (XLSX/XLS), CSV"
        )
        if not uploaded_files:
            st.error("Please upload documents to continue!")
            st.info("💡 **Supported formats**: PDF, TXT, Excel (XLSX/XLS), CSV")
            st.info("📁 **Tip**: You can upload multiple files at once for comprehensive document analysis.")
            st.stop()

        # Display uploaded files info
        st.sidebar.subheader("📎 Uploaded Files")
        for file in uploaded_files:
            file_type = os.path.splitext(file.name)[1].upper()
            st.sidebar.write(f"• {file.name} ({file_type})")

        user_query = st.chat_input(placeholder="Ask me anything about your documents!")

        if uploaded_files and user_query:
            qa_chain = self.setup_qa_chain(uploaded_files)

            utils.display_msg(user_query, 'user')

            with st.chat_message("assistant"):
                st_cb = StreamHandler(st.empty())
                
                # Add a progress indicator
                with st.spinner("🔍 Searching through documents and generating response..."):
                    result = qa_chain.invoke(
                        {"question":user_query},
                        {"callbacks": [st_cb]}
                    )
                
                response = result["answer"]
                st.session_state.messages.append({"role": "assistant", "content": response})
                utils.print_qa(CustomDocChatbot, user_query, response)

                # Enhanced references section
                if result.get('source_documents'):
                    st.markdown("---")
                    st.markdown("### 📖 **References**")
                    
                    for idx, doc in enumerate(result['source_documents'], 1):
                        filename = os.path.basename(doc.metadata['source'])
                        page_num = doc.metadata['page']
                        ref_title = f":blue[Reference {idx}: *{filename} - page.{page_num}*]"
                        with st.popover(ref_title):
                            st.caption(doc.page_content)
                else:
                    st.info("No specific references found in the uploaded documents.")

if __name__ == "__main__":
    obj = CustomDocChatbot()
    obj.main()