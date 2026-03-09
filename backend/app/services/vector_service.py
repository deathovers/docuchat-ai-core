from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from app.core.config import settings
from typing import List, Dict, Any

class VectorService:
    def __init__(self):
        self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self.index = self.pc.Index(settings.PINECONE_INDEX)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    async def upsert_chunks(self, chunks: List[Dict[str, Any]]):
        vectors = []
        for chunk in chunks:
            embedding = self.embeddings.embed_query(chunk["text"])
            vectors.append({
                "id": f"{chunk['metadata']['session_id']}_{chunk['metadata']['file_name']}_{chunk['metadata']['page_number']}_{chunk['metadata']['chunk_index']}",
                "values": embedding,
                "metadata": {
                    "text": chunk["text"],
                    **chunk["metadata"]
                }
            })
        
        # Batch upsert
        self.index.upsert(vectors=vectors)

    async def delete_session_docs(self, session_id: str):
        self.index.delete(filter={"session_id": session_id})

vector_service = VectorService()
