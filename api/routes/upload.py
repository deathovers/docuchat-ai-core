from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import List
import uuid
from api.models.schemas import UploadResponse, FileInfo
from api.services.ingestion import IngestionService

router = APIRouter()
ingestion_service = IngestionService()

@router.post("/upload", response_model=UploadResponse)
async def upload_documents(
    files: List[UploadFile] = File(...),
    session_id: str = Form(...)
):
    processed_files = []
    
    for file in files:
        if not file.filename.endswith(".pdf"):
            continue
            
        try:
            content = await file.read()
            pages = await ingestion_service.process_pdf(content, file.filename, session_id)
            
            processed_files.append(FileInfo(
                id=str(uuid.uuid4()),
                name=file.filename,
                pages=pages
            ))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing {file.filename}: {str(e)}")

    return UploadResponse(status="success", files=processed_files)
