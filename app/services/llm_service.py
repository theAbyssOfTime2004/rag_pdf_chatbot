import ollama
import openai
import logging
from typing import Dict, Any

from app.core.config import settings
from app.utils.prompt_templates import RAG_PROMPT_TEMPLATE

class LLMService:
    """
    Service để tương tác với các Large Language Models (LLMs).
    Hỗ trợ Ollama làm lựa chọn chính và OpenAI làm phương án dự phòng.
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.ollama_client = None
        self.openai_client = None
        self._initialize_clients()

    def _initialize_clients(self):
        """Khởi tạo các client cho Ollama và OpenAI."""
        try:
            # Thử kết nối tới Ollama
            self.ollama_client = ollama.AsyncClient()
            self.logger.info("✅ Ollama client initialized successfully.")
        except Exception as e:
            self.logger.warning(f"⚠️  Could not initialize Ollama client: {e}. Ollama will not be available.")

        if settings.OPENAI_API_KEY and settings.OPENAI_API_KEY != "your_openai_key_here":
            try:
                self.openai_client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
                self.logger.info("✅ OpenAI client initialized successfully.")
            except Exception as e:
                self.logger.warning(f"⚠️  Could not initialize OpenAI client: {e}. OpenAI will not be available.")
        else:
            self.logger.info("ℹ️  OPENAI_API_KEY not set. OpenAI fallback will not be available.")

    async def generate_response(self, question: str, context: str) -> Dict[str, Any]:
        """
        Tạo câu trả lời từ LLM dựa trên câu hỏi và context.
        Ưu tiên sử dụng Ollama, nếu thất bại sẽ chuyển sang OpenAI.
        """
        prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)
        
        # Ưu tiên Ollama
        if self.ollama_client:
            try:
                self.logger.info(f"🤖 Trying to generate response with Ollama model: {settings.LLM_MODEL}")
                response = await self.ollama_client.chat(
                    model=settings.LLM_MODEL,
                    messages=[{'role': 'user', 'content': prompt}]
                )
                return {
                    "answer": response['message']['content'],
                    "model_used": f"ollama/{settings.LLM_MODEL}",
                    "success": True
                }
            except Exception as e:
                self.logger.error(f"❌ Ollama generation failed: {e}. Trying OpenAI as fallback.")

        # Fallback sang OpenAI
        if self.openai_client:
            try:
                self.logger.info("🤖 Falling back to OpenAI model: gpt-3.5-turbo")
                response = await self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}]
                )
                return {
                    "answer": response.choices[0].message.content,
                    "model_used": "openai/gpt-3.5-turbo",
                    "success": True
                }
            except Exception as e:
                self.logger.error(f"❌ OpenAI generation also failed: {e}")

        # Nếu cả hai đều thất bại
        return {
            "answer": "Rất tiếc, tôi không thể tạo câu trả lời vào lúc này. Vui lòng thử lại sau.",
            "model_used": "none",
            "success": False
        }