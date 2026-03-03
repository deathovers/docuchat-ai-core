from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import List
import uuid
from api.models.schemas import UploadResponse, FileInfo
from api.services.ingestion import IngestionService

router = APIRouter()
ingestion_service = IngestionService()

@router.post("/upload", response_model=UploadResponse)
async def upload_documents(
    session_id: str = Form(...),
    files: List[UploadFile] = File(...)
):
    uploaded_files = []
    try:
        for file in files:
            if not file.filename.endswith(".pdf"):
                continue
            
            content = await file.read()
            num_pages = await ingestion_service.process_pdf(content, file.filename, session_id)
            
            uploaded_files.append(FileInfo(
                id=str(uuid.uuid4()),
                name=file.filename,
                pages=num_pages
            ))
        
        return UploadResponse(status="success", files=uploaded_files)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
