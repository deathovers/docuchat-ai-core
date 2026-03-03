from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from api.core.config import settings
from typing import List

class VectorStore:
    def __init__(self):
        self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=settings.OPENAI_API_KEY,
            model="text-embedding-3-small"
        )
        self.index_name = settings.PINECONE_INDEX_NAME
        self.vectorstore = PineconeVectorStore(
            index_name=self.index_name,
            embedding=self.embeddings,
            pinecone_api_key=settings.PINECONE_API_KEY
        )

    def upsert(self, documents):
        self.vectorstore.add_documents(documents)

    def query(self, query: str, session_id: str, top_k: int = 5):
        return self.vectorstore.similarity_search(
            query,
            k=top_k,
            filter={"session_id": session_id}
        )

    def delete_session(self, session_id: str):
        index = self.pc.Index(self.index_name)
        index.delete(filter={"session_id": session_id})
