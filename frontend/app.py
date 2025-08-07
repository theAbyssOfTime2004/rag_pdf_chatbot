import streamlit as st
import sys
import os

# Add parent directory to path để có thể import từ app/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import các components sẽ tạo
from components.sidebar import render_sidebar
from components.upload import render_upload_page
from components.chat import render_chat_page
from components.document_manager import render_document_manager
from utils.api_client import APIClient

# Cấu hình trang
st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .stButton > button {
        border-radius: 10px;
        border: none;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        color: white;
        font-weight: bold;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #1976d2;
    }
    .ai-message {
        background-color: #f3e5f5;
        border-left: 4px solid #7b1fa2;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main app function"""
    
    # Khởi tạo API client
    if 'api_client' not in st.session_state:
        st.session_state.api_client = APIClient()
    
    # Khởi tạo session state variables
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Upload Documents"
    
    if 'chat_session_id' not in st.session_state:
        st.session_state.chat_session_id = None
        
    if 'uploaded_documents' not in st.session_state:
        st.session_state.uploaded_documents = []

    # Header
    st.markdown('<h1 class="main-header">📚 PDF RAG Chatbot</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    render_sidebar()
    
    # Main content area
    if st.session_state.current_page == "Upload Documents":
        render_upload_page()
    elif st.session_state.current_page == "Chat":
        render_chat_page()
    elif st.session_state.current_page == "Document Manager":
        render_document_manager()
    else:
        st.error("Unknown page selected")

if __name__ == "__main__":
    main()