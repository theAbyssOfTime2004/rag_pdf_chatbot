from fastapi import APIRouter

router = APIRouter()

# Import sub-routers
from .documents import router as documents_router
from .chat import router as chat_router

# Include sub-routers
router.include_router(documents_router, prefix="/documents", tags=["Documents"])
router.include_router(chat_router, prefix="/chat", tags=["Chat"])