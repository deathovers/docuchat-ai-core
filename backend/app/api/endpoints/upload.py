from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException, Header
from app.services.document_service import document_service
from app.services.vector_service import vector_service
import uuid

router = APIRouter()

@router.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    session_id: str = Header(None)
):
    if not session_id:
        session_id = str(uuid.uuid4())
    
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    content = await file.read()
    
    # Process in background to keep API responsive
    background_tasks.add_task(
        process_and_store, content, file.filename, session_id
    )

    return {"message": "File uploaded and processing started", "session_id": session_id, "file_name": file.filename}

async def process_and_store(content: bytes, file_name: str, session_id: str):
    chunks = document_service.process_pdf(content, file_name, session_id)
    await vector_service.upsert_chunks(chunks)
