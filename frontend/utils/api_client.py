import requests
import httpx
import streamlit as st
from typing import List, Dict, Any, Optional
import asyncio

class APIClient:
    """Client để giao tiếp với FastAPI backend"""
    
    def __init__(self, base_url: str = "http://localhost:8000/api/v1"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def upload_document(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """Upload tài liệu lên server"""
        try:
            files = {"file": (filename, file_content, "application/pdf")}
            response = self.session.post(f"{self.base_url}/documents/upload", files=files)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Upload failed: {e}")
            return {"error": str(e)}
    
    def get_documents(self) -> List[Dict[str, Any]]:
        """Lấy danh sách tài liệu"""
        try:
            response = self.session.get(f"{self.base_url}/documents")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to fetch documents: {e}")
            return []
    
    def delete_document(self, document_id: int) -> bool:
        """Xóa tài liệu"""
        try:
            response = self.session.delete(f"{self.base_url}/documents/{document_id}")
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to delete document: {e}")
            return False
    
    def ask_question(self, question: str, session_id: str = None, document_id: int = None) -> Dict[str, Any]:
        """Gửi câu hỏi đến RAG pipeline"""
        try:
            payload = {
                "question": question,
                "session_id": session_id,
                "document_id": document_id
            }
            response = self.session.post(f"{self.base_url}/chat/ask", json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to get answer: {e}")
            return {"error": str(e)}
    
    def get_chat_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Lấy lịch sử chat"""
        try:
            response = self.session.get(f"{self.base_url}/chat/sessions/{session_id}/messages")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to fetch chat history: {e}")
            return []
    
    def submit_feedback(self, message_id: int, is_helpful: bool, comment: str = None) -> bool:
        """Gửi feedback"""
        try:
            payload = {
                "message_id": message_id,
                "is_helpful": is_helpful,
                "comment": comment
            }
            response = self.session.post(f"{self.base_url}/chat/feedback", json=payload)
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to submit feedback: {e}")
            return False
    
    def create_chat_session(self, title: str = "New Chat") -> Dict[str, Any]:
        """Tạo phiên chat mới"""
        try:
            payload = {"title": title}
            response = self.session.post(f"{self.base_url}/chat/sessions", json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to create chat session: {e}")
            return {"error": str(e)}