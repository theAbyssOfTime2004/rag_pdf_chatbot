import streamlit as st
from datetime import datetime

def render_sidebar():
    """Render sidebar với navigation và thông tin hệ thống"""
    
    with st.sidebar:
        st.title("🧭 Navigation")
        
        # Menu navigation
        pages = ["Upload Documents", "Chat", "Document Manager"]
        selected_page = st.selectbox("Go to", pages, index=pages.index(st.session_state.current_page))
        
        if selected_page != st.session_state.current_page:
            st.session_state.current_page = selected_page
            st.rerun()
        
        st.divider()
        
        # System status
        st.subheader("🔋 System Status")
        
        # API connection status
        try:
            status = st.session_state.api_client.session.get(
                f"{st.session_state.api_client.base_url.replace('/api/v1', '')}/health", 
                timeout=2
            )
            if status.status_code == 200:
                st.success("✅ Backend Connected")
            else:
                st.error("❌ Backend Error")
        except:
            st.error("❌ Backend Offline")
        
        # Document count
        docs = st.session_state.api_client.get_documents()
        st.info(f"📚 Documents: {len(docs)}")
        
        # Current session info
        if st.session_state.chat_session_id:
            st.info(f"💬 Active Session: {st.session_state.chat_session_id[:8]}...")
        else:
            st.warning("💬 No Active Session")
        
        st.divider()
        
        # Quick actions
        st.subheader("⚡ Quick Actions")
        
        if st.button("🔄 Refresh Data", use_container_width=True):
            # Clear cache và refresh data
            st.cache_data.clear()
            st.rerun()
        
        if st.button("🆕 New Chat Session", use_container_width=True):
            # Tạo session mới
            new_session = st.session_state.api_client.create_chat_session()
            if 'session_id' in new_session:
                st.session_state.chat_session_id = new_session['session_id']
                st.session_state.current_page = "Chat"
                st.success("Created new chat session!")
                st.rerun()
        
        st.divider()
        
        # About section
        st.subheader("ℹ️ About")
        st.caption(f"""
        **PDF RAG Chatbot**  
        Version: 1.0.0  
        Built with Streamlit & FastAPI  
        Time: {datetime.now().strftime('%H:%M:%S')}
        """)