import streamlit as st
from typing import List
import time

def render_upload_page():
    """Render trang upload tài liệu"""
    
    st.header("📄 Upload Documents")
    st.write("Upload your PDF documents to start chatting with them!")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload one or more PDF files (max 50MB each)"
    )
    
    # Upload button và xử lý
    if uploaded_files:
        if st.button("🚀 Process Documents", type="primary", use_container_width=True):
            process_uploads(uploaded_files)
    
    # Hiển thị danh sách tài liệu đã upload
    st.divider()
    display_uploaded_documents()

def process_uploads(uploaded_files: List):
    """Xử lý việc upload các file"""
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_files = len(uploaded_files)
    
    for i, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Processing {uploaded_file.name}...")
        
        try:
            # Đọc file content
            file_content = uploaded_file.read()
            
            # Upload lên server
            result = st.session_state.api_client.upload_document(
                file_content=file_content,
                filename=uploaded_file.name
            )
            
            if "error" not in result:
                st.success(f"✅ {uploaded_file.name} uploaded successfully!")
                
                # Hiển thị thông tin chi tiết
                with st.expander(f"📋 Details for {uploaded_file.name}"):
                    st.json(result)
                    
            else:
                st.error(f"❌ Failed to upload {uploaded_file.name}: {result['error']}")
                
        except Exception as e:
            st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
        
        # Cập nhật progress
        progress = (i + 1) / total_files
        progress_bar.progress(progress)
    
    status_text.text("✅ All files processed!")
    time.sleep(1)
    
    # Clear progress sau khi hoàn thành
    progress_bar.empty()
    status_text.empty()
    
    # Refresh document list
    st.rerun()

@st.cache_data(ttl=30)
def get_documents_cached():
    """Cache danh sách documents để tránh call API liên tục"""
    return st.session_state.api_client.get_documents()

def display_uploaded_documents():
    """Hiển thị danh sách tài liệu đã upload"""
    
    st.subheader("📚 Your Documents")
    
    documents = get_documents_cached()
    
    if not documents:
        st.info("No documents uploaded yet. Upload some PDFs to get started!")
        return
    
    # Hiển thị documents trong dạng cards
    for doc in documents:
        with st.container():
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.write(f"**📄 {doc.get('filename', 'Unknown')}**")
                st.caption(f"Uploaded: {doc.get('created_at', 'Unknown')}")
                if doc.get('page_count'):
                    st.caption(f"Pages: {doc['page_count']}")
            
            with col2:
                # Status badge
                status = doc.get('processing_status', 'unknown')
                if status == 'completed':
                    st.success("✅ Ready")
                elif status == 'processing':
                    st.warning("⏳ Processing")
                elif status == 'failed':
                    st.error("❌ Failed")
                else:
                    st.info("❓ Unknown")
            
            with col3:
                # Actions
                if st.button(f"🗑️ Delete", key=f"delete_{doc.get('id')}", help="Delete this document"):
                    delete_document(doc.get('id'))
            
            st.divider()

def delete_document(document_id: int):
    """Xóa tài liệu"""
    
    if st.session_state.api_client.delete_document(document_id):
        st.success("Document deleted successfully!")
        # Clear cache và refresh
        st.cache_data.clear()
        st.rerun()
    else:
        st.error("Failed to delete document")