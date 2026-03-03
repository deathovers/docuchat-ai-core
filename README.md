# DocuChat AI

DocuChat AI is a Retrieval-Augmented Generation (RAG) application designed to allow users to interact with multiple PDF documents simultaneously. By leveraging advanced LLMs and vector databases, it provides grounded, citation-backed answers to user queries based strictly on the uploaded content.

## 1. Project Overview
DocuChat AI processes PDF documents, indexes their content into a vector database, and provides a conversational interface. It ensures data privacy through session-based isolation and maintains high accuracy by enforcing strict grounding in the provided context.

## 2. Key Features
- **Multi-PDF Support:** Upload and query across multiple documents at once.
- **Grounded Responses:** Answers are derived strictly from the provided context to prevent hallucinations.
- **Automatic Citations:** Every answer includes references to the specific document and page number.
- **Session Isolation:** User data is isolated using session IDs and metadata filtering.
- **Asynchronous Pipeline:** Built with FastAPI and AsyncOpenAI for high performance.

## 3. System Architecture
The system follows a standard RAG architecture:
1. **Ingestion Pipeline:** PDF Parsing (PyMuPDF) -> Recursive Chunking -> Embedding Generation (OpenAI `text-embedding-3-small`) -> Vector Storage (Pinecone).
2. **Retrieval Pipeline:** Query Embedding -> Similarity Search (filtered by `session_id`) -> Context Retrieval.
3. **Generation Pipeline:** Prompt Construction -> LLM Processing (GPT-4o) -> Response with Citations.

## 4. Installation

### Prerequisites
- Python 3.9+
- OpenAI API Key
- Pinecone API Key and Environment

### Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-repo/docuchat-ai.git
   cd docuchat-ai
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables:**
   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=your_openai_key
   PINECONE_API_KEY=your_pinecone_key
   PINECONE_ENVIRONMENT=your_environment
   PINECONE_INDEX_NAME=docuchat-index
   ```

## 5. Usage

### Running the Server
```bash
uvicorn api.main:app --reload
```

### Basic Workflow
1. **Upload Documents:** Send a `POST` request to `/v1/upload` with your PDF files and a `session_id`.
2. **Chat:** Send a `POST` request to `/v1/chat` with your query and the same `session_id`.
3. **Manage:** Use `DELETE /v1/document/{id}` to remove specific files from your session.

## 6. Technical Constraints
- **Format:** Strictly PDF support (No OCR for scanned images in MVP).
- **Persistence:** Session-scoped data; vectors may be purged after 24 hours of inactivity.
- **Performance:** Designed for < 5s end-to-end latency.
