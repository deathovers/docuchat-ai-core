import fitz  # PyMuPDF
from typing import List, Dict
from langchain.schema import Document

class PDFService:
    @staticmethod
    def extract_text_with_metadata(file_content: bytes, filename: str) -> List[Document]:
        doc = fitz.open(stream=file_content, filetype="pdf")
        documents = []
        
        for page_num, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                metadata = {
                    "doc_name": filename,
                    "page_label": page_num + 1
                }
                documents.append(Document(page_content=text, metadata=metadata))
        
        return documents
