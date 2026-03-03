from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import List
from app.models.schemas import ChatQueryRequest, ChatQueryResponse, UploadResponse
from app.services.pdf_service import PDFService
from app.services.vector_service import VectorService
from app.services.rag_service import RAGService

router = APIRouter()
vector_service = VectorService()
pdf_service = PDFService()
rag_service = RAGService(vector_service)

@router.post("/documents/upload", response_model=UploadResponse)
async def upload_documents(
    session_id: str = Form(...),
    files: List[UploadFile] = File(...)
):
    file_ids = []
    for file in files:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"File {file.filename} is not a PDF.")
        
        content = await file.read()
        documents = pdf_service.extract_text_with_metadata(content, file.filename)
        
        if documents:
            vector_service.add_documents(session_id, documents)
            file_ids.append(file.filename)
            
    return UploadResponse(file_ids=file_ids, status="success")

@router.post("/chat/query", response_model=ChatQueryResponse)
async def chat_query(request: ChatQueryRequest):
    try:
        response = await rag_service.answer_query(request.session_id, request.query)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/documents/{session_id}")
async def clear_session(session_id: str):
    vector_service.delete_session_data(session_id)
    return {"status": "session data cleared"}
