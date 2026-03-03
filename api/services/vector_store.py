import logging
from pinecone import Pinecone
from api.core.config import settings

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self):
        try:
            self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
            self.index = self.pc.Index(settings.PINECONE_INDEX)
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            raise

    def upsert_vectors(self, vectors: List[Dict]):
        """
        vectors: List of tuples (id, vector, metadata)
        """
        try:
            self.index.upsert(vectors=vectors)
        except Exception as e:
            logger.error(f"Pinecone upsert error: {e}")
            raise

    def query_vectors(self, vector: List[float], session_id: str, top_k: int = 5):
        try:
            return self.index.query(
                vector=vector,
                filter={"session_id": {"$eq": session_id}},
                top_k=top_k,
                include_metadata=True
            )
        except Exception as e:
            logger.error(f"Pinecone query error: {e}")
            raise

    def delete_by_session(self, session_id: str):
        try:
            self.index.delete(filter={"session_id": {"$eq": session_id}})
        except Exception as e:
            logger.error(f"Pinecone delete error: {e}")
            raise
