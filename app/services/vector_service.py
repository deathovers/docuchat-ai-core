import chromadb
from chromadb.utils import embedding_functions
from app.core.config import settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List

class VectorService:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIRECTORY)
        self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=settings.OPENAI_API_KEY,
            model_name=settings.EMBEDDING_MODEL
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )

    def get_collection(self, session_id: str):
        # Session isolation: each session gets its own collection
        return self.client.get_or_create_collection(
            name=f"session_{session_id}",
            embedding_function=self.embedding_fn
        )

    def add_documents(self, session_id: str, documents: List[Document]):
        collection = self.get_collection(session_id)
        
        # Split documents into chunks
        chunks = self.text_splitter.split_documents(documents)
        
        ids = [f"{session_id}_{i}_{hash(chunk.page_content)}" for i, chunk in enumerate(chunks)]
        metadatas = [chunk.metadata for chunk in chunks]
        contents = [chunk.page_content for chunk in chunks]
        
        collection.add(
            ids=ids,
            metadatas=metadatas,
            documents=contents
        )

    def query(self, session_id: str, query_text: str, n_results: int = 5):
        collection = self.get_collection(session_id)
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        return results

    def delete_session_data(self, session_id: str):
        try:
            self.client.delete_collection(name=f"session_{session_id}")
        except Exception:
            pass
