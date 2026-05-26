# Enterprise Knowledge Assistant

A RAG-based internal AI assistant for education/training institutes. Users can upload documents, store them in a vector database, and ask questions based on those documents.

## Tech Stack

- **Backend**: Python, FastAPI
- **Vector Database**: ChromaDB
- **Frontend**: Streamlit
- **Embeddings**: Sentence Transformers

## Project Structure

```
enterprise-knowledge-assistant/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI application entry point
│   │   ├── config.py            # Configuration settings
│   │   ├── routes/
│   │   │   ├── upload_routes.py # Document upload endpoints
│   │   │   └── chat_routes.py   # Chat/query endpoints
│   │   ├── services/
│   │   │   ├── document_loader.py   # Load various document formats
│   │   │   ├── chunking_service.py  # Split documents into chunks
│   │   │   ├── embedding_service.py # Generate embeddings
│   │   │   ├── vector_store.py      # ChromaDB operations
│   │   │   └── rag_service.py       # RAG orchestration
│   │   ├── models/
│   │   │   └── schemas.py       # Pydantic models
│   │   └── utils/
│   │       └── file_utils.py    # File utilities
│   ├── data/
│   │   ├── raw/                 # Original uploaded documents
│   │   └── processed/           # Processed document data
│   ├── vector_db/               # ChromaDB persistence
│   └── requirements.txt
├── frontend/
│   ├── streamlit_app.py         # Streamlit UI
│   └── requirements.txt
├── sample_documents/            # Sample documents for testing
├── .gitignore
└── README.md
```

## Setup

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Frontend

```bash
cd frontend
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## API Endpoints

- `GET /health` - Health check
- `POST /upload/document` - Upload a document
- `POST /chat/query` - Query the knowledge base

## License

MIT
