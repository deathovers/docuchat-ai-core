import fitz  # PyMuPDF
import uuid
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from api.core.config import settings
from api.services.vector_store import VectorStore

class IngestionService:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        self.vector_store = VectorStore()

    async def process_pdf(self, file_bytes: bytes, file_name: str, session_id: str) -> int:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        total_pages = len(doc)
        documents = []

        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            if not text.strip():
                continue
                
            chunks = self.text_splitter.split_text(text)
            for chunk in chunks:
                documents.append(Document(
                    page_content=chunk,
                    metadata={
                        "file_id": str(uuid.uuid4()),
                        "file_name": file_name,
                        "page_number": page_num,
                        "session_id": session_id
                    }
                ))
        
        if documents:
            self.vector_store.upsert(documents)
            
        return total_pages
