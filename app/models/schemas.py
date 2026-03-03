from pydantic import BaseModel
from typing import List, Optional

class ChatQueryRequest(BaseModel):
    session_id: str
    query: str
    history: Optional[List[dict]] = []

class Source(BaseModel):
    document_name: str
    page: int

class ChatQueryResponse(BaseModel):
    answer: str
    sources: List[Source]

class UploadResponse(BaseModel):
    file_ids: List[str]
    status: str
