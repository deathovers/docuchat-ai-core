from pydantic import BaseModel
from typing import List, Optional

class Source(BaseModel):
    document: str
    page: int
    snippet: str

class ChatRequest(BaseModel):
    session_id: str
    query: str
    stream: bool = False

class ChatResponse(BaseModel):
    answer: str
    sources: List[Source]

class FileInfo(BaseModel):
    id: str
    name: str
    pages: int

class UploadResponse(BaseModel):
    status: str
    files: List[FileInfo]
