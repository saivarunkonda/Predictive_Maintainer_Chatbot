import os
import openai
import streamlit as st
from datetime import datetime
from streamlit.logger import get_logger
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings, OllamaEmbeddings
# For Gemini AI:
from langchain_google_genai import ChatGoogleGenerativeAI

logger = get_logger('Langchain-Chatbot')

#decorator
def enable_chat_history(func):
    if os.environ.get("OPENAI_API_KEY"):

        # to clear chat history after swtching chatbot
        current_page = func.__qualname__
        if "current_page" not in st.session_state:
            st.session_state["current_page"] = current_page
        if st.session_state["current_page"] != current_page:
            try:
                st.cache_resource.clear()
                del st.session_state["current_page"]
                del st.session_state["messages"]
            except:
                pass

        # to show chat history on ui
        if "messages" not in st.session_state:
            st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
        for msg in st.session_state["messages"]:
            st.chat_message(msg["role"]).write(msg["content"])

    def execute(*args, **kwargs):
        func(*args, **kwargs)
    return execute

def display_msg(msg, author):
    """Method to display message on the UI

    Args:
        msg (str): message to display
        author (str): author of the message -user/assistant
    """
    st.session_state.messages.append({"role": author, "content": msg})
    st.chat_message(author).write(msg)

def choose_custom_openai_key():
    openai_api_key = st.sidebar.text_input(
        label="OpenAI API Key",
        type="password",
        placeholder="sk-...",
        key="SELECTED_OPENAI_API_KEY"
        )
    if not openai_api_key:
        st.error("Please add your OpenAI API key to continue.")
        st.info("Obtain your key from this link: https://platform.openai.com/account/api-keys")
        st.stop()

    model = "gpt-4.1-mini"
    try:
        client = openai.OpenAI(api_key=openai_api_key)
        available_models = [{"id": i.id, "created":datetime.fromtimestamp(i.created)} for i in client.models.list() if str(i.id).startswith("gpt")]
        available_models = sorted(available_models, key=lambda x: x["created"])
        available_models = [i["id"] for i in available_models]

        model = st.sidebar.selectbox(
            label="Model",
            options=available_models,
            key="SELECTED_OPENAI_MODEL"
        )
    except openai.AuthenticationError as e:
        st.error(e.body["message"])
        st.stop()
    except Exception as e:
        print(e)
        st.error("Something went wrong. Please try again later.")
        st.stop()
    return model, openai_api_key

def configure_llm():
    available_llms = ["gpt-4.1-mini","llama3.2:3b","gemini-1.5-pro","use your openai api key"]
    llm_opt = st.sidebar.radio(
        label="LLM",
        options=available_llms,
        key="SELECTED_LLM"
        )

    if llm_opt == "llama3.2:3b":
        llm = ChatOllama(
            model="llama3.2:3b", 
            base_url=st.secrets["OLLAMA_ENDPOINT"],
            temperature=0,
            # Force CPU usage and optimize memory
            num_ctx=2048,  # Reduce context window
            num_gpu=0,     # Force CPU usage
            num_thread=4   # Limit CPU threads
        )
    elif llm_opt == "gpt-4.1-mini":
        llm = ChatOpenAI(model_name=llm_opt, temperature=0, streaming=True, api_key=st.secrets["OPENAI_API_KEY"])
    elif llm_opt == "gemini-1.5-pro":
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0, streaming=True, google_api_key=st.secrets["GOOGLE_API_KEY"])
    else:
        model, openai_api_key = choose_custom_openai_key()
        llm = ChatOpenAI(model_name=model, temperature=0, streaming=True, api_key=openai_api_key)
        
        # Gemini equivalent (uncomment to use):
        # llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0, streaming=True, google_api_key=st.secrets["GOOGLE_API_KEY"])
    return llm

def print_qa(cls, question, answer):
    log_str = "\nUsecase: {}\nQuestion: {}\nAnswer: {}\n" + "------"*10
    logger.info(log_str.format(cls.__name__, question, answer))

@st.cache_resource
def configure_embedding_model():
    """Configure embedding model using Ollama (local and free)"""
    try:
        # First try Ollama embeddings using the same model as the LLM
        ollama_endpoint = st.secrets.get("OLLAMA_ENDPOINT", "http://localhost:11434")
        embedding_model = OllamaEmbeddings(
            base_url=ollama_endpoint,
            model="llama3.2:3b"  # Use the same model for embeddings
        )
        
        # Test with a small document to ensure it works
        test_embed = embedding_model.embed_query("test")
        if test_embed:
            st.success("✅ Ollama embeddings loaded successfully (Local & Free)")
            return embedding_model
            
    except Exception as e:
        st.warning(f"⚠️ Ollama embeddings failed: {str(e)[:100]}...")
        st.info("🔄 Switching to HuggingFace embeddings...")
        
        try:
            # Fallback to a lightweight HuggingFace model (also free and local)
            embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={
                    'device': 'cpu',
                    'trust_remote_code': True
                },
                encode_kwargs={
                    'normalize_embeddings': False,
                    'batch_size': 1,  # Process one at a time to save memory
                    'show_progress_bar': False
                }
            )
            # Test the model
            test_embed = embedding_model.embed_query("test")
            if test_embed:
                st.success("✅ HuggingFace embeddings loaded successfully (Local & Free)")
                return embedding_model
                
        except Exception as e2:
            st.warning(f"⚠️ HuggingFace embeddings failed: {str(e2)[:100]}...")
            st.info("🔄 Trying FastEmbed as final fallback...")
            
            try:
                # Final fallback to FastEmbed (also free and local)
                embedding_model = FastEmbedEmbeddings(
                    model_name="BAAI/bge-small-en",  # Smaller model
                    max_length=512,  # Limit sequence length
                    doc_embed_type="default"
                )
                test_embed = embedding_model.embed_query("test")
                if test_embed:
                    st.success("✅ FastEmbed model loaded successfully (Local & Free)")
                    return embedding_model
                    
            except Exception as e3:
                st.error(f"❌ All local embedding options failed: {str(e3)}")
                st.error("**Solutions:**")
                st.error("1. ✅ Ensure Ollama is running: `ollama serve`")
                st.error("2. ✅ Install sentence-transformers: `pip install sentence-transformers`")
                st.error("3. ✅ Install fastembed: `pip install fastembed`")
                st.error("4. ✅ Reduce document size or restart the application")
                st.info("💡 **Note**: This app uses only FREE local models - no API keys required!")
                st.stop()

def sync_st_session():
    for k, v in st.session_state.items():
        st.session_state[k] = v
