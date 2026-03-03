# API Documentation: DocuChat AI

The DocuChat AI API provides endpoints for document ingestion, conversational retrieval, and document management.

## Base URL
`http://localhost:8000/v1`

---

## Endpoints

### 1. Upload Documents
`POST /upload`

Uploads one or more PDF files to be processed and indexed for a specific session.

**Request:**
- **Type:** `multipart/form-data`
- **Parameters:**
    - `files`: List of PDF files (Max 20MB per file).
    - `session_id`: A unique string identifying the user session.

**Response:**
```json
{
  "status": "success",
  "files": [
    {
      "id": "uuid-123",
      "name": "annual_report.pdf",
      "pages": 15
    }
  ]
}
```

### 2. Chat / Query
`POST /chat`

Submits a natural language query and retrieves a grounded response based on the uploaded documents.

**Request Body:**
```json
{
  "session_id": "uuid-123",
  "query": "What was the total revenue in 2023?",
  "stream": false
}
```

**Response Body:**
```json
{
  "answer": "The total revenue in 2023 was $5.2 billion [annual_report.pdf - Page 4].",
  "sources": [
    {
      "document": "annual_report.pdf",
      "page": 4,
      "snippet": "...total revenue for the fiscal year 2023 reached $5.2 billion..."
    }
  ]
}
```

### 3. Delete Document
`DELETE /document/{id}`

Removes a specific document and its associated vectors from the session.

**Parameters:**
- `id`: The unique ID of the document returned during upload.

**Response:**
```json
{
  "status": "deleted",
  "document_id": "uuid-123"
}
```

---

## Data Models

### ChatResponse
| Field | Type | Description |
| :--- | :--- | :--- |
| `answer` | string | The generated answer from the LLM. |
| `sources` | array | List of source objects used for the answer. |

### Source
| Field | Type | Description |
| :--- | :--- | :--- |
| `document` | string | Filename of the source document. |
| `page` | integer | Page number where the info was found. |
| `snippet` | string | A short text excerpt from the document. |
