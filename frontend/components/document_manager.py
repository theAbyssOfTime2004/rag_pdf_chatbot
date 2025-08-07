import streamlit as st
from typing import List, Dict, Any
import json
import time

def render_document_manager():
    """Render trang quản lý tài liệu"""
    
    st.header("🗂️ Document Manager")
    st.write("Manage your uploaded documents and view detailed information.")
    
    # Refresh button
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    # Documents overview
    display_documents_overview()
    
    # Detailed document list
    display_detailed_documents()

def display_documents_overview():
    """Hiển thị tổng quan tài liệu"""
    
    documents = st.session_state.api_client.get_documents()
    
    # Statistics
    total_docs = len(documents)
    completed_docs = len([d for d in documents if d.get('processing_status') == 'completed'])
    processing_docs = len([d for d in documents if d.get('processing_status') == 'processing'])
    failed_docs = len([d for d in documents if d.get('processing_status') == 'failed'])
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📚 Total Documents", total_docs)
    
    with col2:
        st.metric("✅ Ready", completed_docs)
    
    with col3:
        st.metric("⏳ Processing", processing_docs)
    
    with col4:
        st.metric("❌ Failed", failed_docs)
    
    st.divider()

def display_detailed_documents():
    """Hiển thị danh sách chi tiết tài liệu"""
    
    documents = st.session_state.api_client.get_documents()
    
    if not documents:
        st.info("📝 No documents found. Go to Upload page to add some documents.")
        return
    
    # Filter và search
    search_term = st.text_input("🔍 Search documents:", placeholder="Enter filename...")
    
    status_filter = st.selectbox(
        "Filter by status:",
        options=["All", "completed", "processing", "failed"]
    )
    
    # Apply filters
    filtered_docs = documents
    if search_term:
        filtered_docs = [d for d in filtered_docs if search_term.lower() in d.get('filename', '').lower()]
    
    if status_filter != "All":
        filtered_docs = [d for d in filtered_docs if d.get('processing_status') == status_filter]
    
    st.write(f"Showing {len(filtered_docs)} of {len(documents)} documents")
    
    # Document cards
    for doc in filtered_docs:
        render_document_card(doc)

def render_document_card(doc: Dict[str, Any]):
    """Render card cho một document"""
    
    with st.container():
        # Header with filename và status
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.subheader(f"📄 {doc.get('filename', 'Unknown')}")
        
        with col2:
            status = doc.get('processing_status', 'unknown')
            if status == 'completed':
                st.success("✅ Ready")
            elif status == 'processing':
                st.warning("⏳ Processing")
            elif status == 'failed':
                st.error("❌ Failed")
            elif status == 'pending':
                st.warning("⏸️ Pending")
            else:
                st.info("❓ Unknown")
        
        with col3:
            # Thêm nút Retry nếu status là pending
            if status == 'pending':
                if st.button(f"🔄 Retry", key=f"retry_{doc.get('id')}", type="primary"):
                    retry_document_processing(doc.get('id'))
                    
            # Nút Delete
            if st.button(f"🗑️ Delete", key=f"delete_detail_{doc.get('id')}", type="secondary"):
                delete_document_with_confirmation(doc)
        
        # Document details
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Basic Information:**")
            st.write(f"- **ID:** {doc.get('id', 'N/A')}")
            st.write(f"- **Filename:** {doc.get('filename', 'N/A')}")
            st.write(f"- **File Size:** {format_file_size(doc.get('file_size', 0))}")
            st.write(f"- **Upload Time:** {doc.get('created_at', 'N/A')}")
        
        with col2:
            st.write("**Processing Information:**")
            st.write(f"- **Status:** {doc.get('processing_status', 'N/A')}")
            st.write(f"- **Pages:** {doc.get('page_count', 'N/A')}")
            st.write(f"- **Chunks:** {doc.get('chunk_count', 'N/A')}")
            
            if doc.get('processing_error'):
                st.error(f"Error: {doc['processing_error']}")
        
        # Detailed metadata
        if st.checkbox(f"Show raw metadata", key=f"metadata_{doc.get('id')}"):
            st.json(doc)
        
        st.divider()

def delete_document_with_confirmation(doc: Dict[str, Any]):
    """Xóa document với confirmation"""
    
    # Sử dụng dialog hoặc session state cho confirmation
    confirm_key = f"confirm_delete_{doc.get('id')}"
    
    if confirm_key not in st.session_state:
        st.session_state[confirm_key] = False
    
    if not st.session_state[confirm_key]:
        st.warning(f"⚠️ Are you sure you want to delete '{doc.get('filename')}'?")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Yes, Delete", key=f"confirm_yes_{doc.get('id')}", type="primary"):
                st.session_state[confirm_key] = True
                st.rerun()
        
        with col2:
            if st.button("❌ Cancel", key=f"confirm_no_{doc.get('id')}"):
                st.info("Deletion cancelled")
    else:
        # Perform actual deletion
        if st.session_state.api_client.delete_document(doc.get('id')):
            st.success(f"✅ '{doc.get('filename')}' deleted successfully!")
            del st.session_state[confirm_key]
            st.cache_data.clear()
            st.rerun()
        else:
            st.error("❌ Failed to delete document")
            del st.session_state[confirm_key]

def retry_document_processing(document_id: int):
    """Retry processing cho document"""
    try:
        # Sử dụng method mới trong API client
        result = st.session_state.api_client.trigger_document_processing(document_id)
        
        if "error" not in result:
            st.success("🔄 Processing triggered! Check status in a few moments.")
            time.sleep(2)
            st.cache_data.clear()
            st.rerun()
        else:
            st.error(f"❌ Failed to trigger processing: {result['error']}")
    except Exception as e:
        st.error(f"❌ Error: {e}")

def format_file_size(size_bytes: int) -> str:
    """Format file size thành dạng human readable"""
    
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"