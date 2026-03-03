import fitz  # PyMuPDF
import uuid
import logging
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import AsyncOpenAI
from api.core.config import settings
from api.services.vector_store import VectorStore

logger = logging.getLogger(__name__)

class IngestionService:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.vector_store = VectorStore()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

    async def process_pdf(self, file_bytes: bytes, filename: str, session_id: str):
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            num_pages = len(doc)
            
            all_chunks = []
            
            for page_num in range(num_pages):
                page = doc.load_page(page_num)
                text = page.get_text()
                
                chunks = self.text_splitter.split_text(text)
                for chunk in chunks:
                    all_chunks.append({
                        "text": chunk,
                        "metadata": {
                            "file_name": filename,
                            "page_number": page_num + 1,
                            "session_id": session_id
                        }
                    })
            
            # Generate embeddings and upsert
            vectors_to_upsert = []
            for i, chunk_data in enumerate(all_chunks):
                embed_res = await self.client.embeddings.create(
                    input=chunk_data["text"],
                    model="text-embedding-3-small"
                )
                vector = embed_res.data[0].embedding
                
                # Add text to metadata for retrieval
                metadata = chunk_data["metadata"]
                metadata["text"] = chunk_data["text"]
                
                vectors_to_upsert.append((
                    f"{session_id}_{uuid.uuid4()}",
                    vector,
                    metadata
                ))
            
            if vectors_to_upsert:
                self.vector_store.upsert_vectors(vectors_to_upsert)
            
            return num_pages
        except Exception as e:
            logger.error(f"Ingestion error for {filename}: {e}")
            raise
