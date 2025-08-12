# RAG PDF Chatbot

A complete RAG (Retrieval-Augmented Generation) system for PDF document Q&A using FastAPI, Streamlit, and PostgreSQL.

##  Quick Start with Docker

### Prerequisites
- Docker and Docker Compose installed
- Git

### Setup & Run
```bash
# 1. Clone repository
git clone <your-repo-url>
cd rag_pdf_chatbot

# 2. Start the entire stack
docker compose up --build --remove-orphans --env-file .env.docker

# 3. Access applications
# Frontend (Streamlit UI): http://localhost:8501
# Backend (FastAPI): http://localhost:8000
# API Documentation: http://localhost:8000/docs
```

### Stop the application
```bash
docker compose down
```

##  Project Structure
```
rag_pdf_chatbot/
├── app/                    # FastAPI backend
│   ├── api/               # API routes
│   ├── models/            # Database models
│   ├── services/          # Business logic
│   └── main.py           # FastAPI application
├── frontend/              # Streamlit frontend
│   ├── app.py            # Main Streamlit app
│   └── requirements.txt  # Frontend dependencies
├── alembic/              # Database migrations
├── docker-compose.yml    # Docker services configuration
├── Dockerfile           # Backend container
├── .env.docker          # Docker environment variables
└── requirements.txt     # Backend dependencies
```

##  Tech Stack
- **Backend:** FastAPI, SQLAlchemy, Alembic
- **Frontend:** Streamlit
- **Database:** PostgreSQL
- **AI/ML:** Sentence Transformers, FAISS, LangChain
- **LLM:** Ollama (configurable)
- **Deployment:** Docker & Docker Compose

##  Features
- PDF document upload and processing
- Text extraction with OCR support
- Vector embedding and similarity search
- RAG-based question answering
- Real-time chat interface
- Persistent conversation history

##  Development Mode (Optional)
If you want to run components separately for development:

```bash
# Terminal 1: Start local PostgreSQL
sudo systemctl start postgresql

# Terminal 2: Backend with hot reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 3: Frontend with hot reload
streamlit run frontend/app.py --server.port 8501
```

