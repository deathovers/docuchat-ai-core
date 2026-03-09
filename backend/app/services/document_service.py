import fitz
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentService:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )

    def process_pdf(self, file_content: bytes, file_name: str, session_id: str) -> List[Dict[str, Any]]:
        doc = fitz.open(stream=file_content, filetype="pdf")
        chunks = []
        
        for page_num, page in enumerate(doc):
            text = page.get_text()
            if not text.strip():
                continue
                
            page_chunks = self.text_splitter.split_text(text)
            for i, chunk_text in enumerate(page_chunks):
                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        "session_id": session_id,
                        "file_name": file_name,
                        "page_number": page_num + 1,
                        "chunk_index": i
                    }
                })
        
        return chunks

document_service = DocumentService()
