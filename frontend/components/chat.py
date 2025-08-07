import streamlit as st
from typing import List, Dict, Any
import uuid
import time

def render_chat_page():
    """Render trang chat"""
    
    st.header("💬 Chat with Your Documents")
    
    # Kiểm tra có documents không
    documents = st.session_state.api_client.get_documents()
    if not documents:
        st.warning("⚠️ No documents found. Please upload some documents first!")
        if st.button("Go to Upload Page"):
            st.session_state.current_page = "Upload Documents"
            st.rerun()
        return
    
    # Tạo hoặc chọn chat session
    setup_chat_session()
    
    # Hiển thị chat history
    display_chat_history()
    
    # Chat input
    chat_input_area()

def setup_chat_session():
    """Setup chat session"""
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if not st.session_state.chat_session_id:
            st.info("🆕 No active chat session. Create one to start chatting!")
    
    with col2:
        if st.button("🆕 New Session", use_container_width=True):
            create_new_session()

def create_new_session():
    """Tạo session chat mới"""
    
    new_session = st.session_state.api_client.create_chat_session("New Chat Session")
    
    if 'session_id' in new_session:
        st.session_state.chat_session_id = new_session['session_id']
        st.success("✅ New chat session created!")
        st.rerun()
    else:
        st.error("❌ Failed to create chat session")

def display_chat_history():
    """Hiển thị lịch sử chat"""
    
    if not st.session_state.chat_session_id:
        return
    
    # Container cho chat history
    chat_container = st.container()
    
    with chat_container:
        messages = st.session_state.api_client.get_chat_history(st.session_state.chat_session_id)
        
        if not messages:
            st.info("👋 Start a conversation by asking a question about your documents!")
            return
        
        # Hiển thị messages
        for msg in messages:
            if msg['message_type'] == 'user':
                render_user_message(msg)
            elif msg['message_type'] == 'ai':
                render_ai_message(msg)

def render_user_message(message: Dict[str, Any]):
    """Render tin nhắn của user"""
    
    with st.chat_message("user", avatar="👤"):
        st.write(message['content'])
        st.caption(f"🕐 {message.get('created_at', 'Unknown time')}")

def render_ai_message(message: Dict[str, Any]):
    """Render tin nhắn của AI"""
    
    with st.chat_message("assistant", avatar="🤖"):
        st.write(message['content'])
        
        # Hiển thị model info
        metadata = message.get('message_metadata', {})
        if metadata.get('model_used'):
            st.caption(f"🧠 Model: {metadata['model_used']}")
        
        # Hiển thị sources nếu có
        if metadata.get('sources'):
            render_sources(metadata['sources'])
        
        # Feedback buttons
        render_feedback_buttons(message['id'])
        
        st.caption(f"🕐 {message.get('created_at', 'Unknown time')}")

def render_sources(sources: List[Dict]):
    """Hiển thị sources của câu trả lời"""
    
    with st.expander(f"📚 Sources ({len(sources)} found)"):
        for i, source in enumerate(sources):
            st.write(f"**Source {i+1}:**")
            st.write(f"- Document ID: {source.get('document_id')}")
            st.write(f"- Page: {source.get('page_number')}")
            st.write(f"- Similarity: {source.get('similarity_score', 0):.3f}")
            
            # Hiển thị đoạn text (rút gọn)
            chunk_text = source.get('chunk_text', '')
            if len(chunk_text) > 200:
                chunk_text = chunk_text[:200] + "..."
            st.text_area(f"Content preview {i+1}:", chunk_text, height=100, key=f"source_{i}")
            st.divider()

def render_feedback_buttons(message_id: int):
    """Render nút feedback"""
    
    col1, col2, col3 = st.columns([1, 1, 4])
    
    with col1:
        if st.button("👍", key=f"helpful_{message_id}", help="This answer was helpful"):
            submit_feedback(message_id, True)
    
    with col2:
        if st.button("👎", key=f"not_helpful_{message_id}", help="This answer was not helpful"):
            submit_feedback(message_id, False)

def submit_feedback(message_id: int, is_helpful: bool):
    """Gửi feedback"""
    
    if st.session_state.api_client.submit_feedback(message_id, is_helpful):
        emoji = "👍" if is_helpful else "👎"
        st.success(f"{emoji} Feedback submitted!")
    else:
        st.error("❌ Failed to submit feedback")

def chat_input_area():
    """Khu vực nhập câu hỏi"""
    
    if not st.session_state.chat_session_id:
        st.warning("⚠️ Please create a chat session first")
        return
    
    # Document selector (optional)
    documents = st.session_state.api_client.get_documents()
    
    with st.expander("🎯 Advanced Options", expanded=False):
        selected_doc = st.selectbox(
            "Search in specific document (optional):",
            options=[None] + [doc['id'] for doc in documents],
            format_func=lambda x: "All documents" if x is None else next(
                (doc['filename'] for doc in documents if doc['id'] == x), f"Document {x}"
            )
        )
    
    # Chat input
    with st.form(key="chat_form", clear_on_submit=True):
        question = st.text_area(
            "Ask a question about your documents:",
            placeholder="What is this document about?",
            height=100
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            submit_button = st.form_submit_button("🚀 Ask", type="primary", use_container_width=True)
    
    # Xử lý submit
    if submit_button and question:
        process_question(question, selected_doc)

def process_question(question: str, document_id: int = None):
    """Xử lý câu hỏi"""
    
    # Hiển thị loading
    with st.spinner("🤔 Thinking..."):
        response = st.session_state.api_client.ask_question(
            question=question,
            session_id=st.session_state.chat_session_id,
            document_id=document_id
        )
    
    if "error" not in response:
        st.success("✅ Got response!")
        # Refresh để hiển thị câu trả lời mới
        st.rerun()
    else:
        st.error(f"❌ Error: {response['error']}")