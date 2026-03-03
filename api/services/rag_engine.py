import logging
from typing import List
from openai import AsyncOpenAI
from api.core.config import settings
from api.services.vector_store import VectorStore
from api.models.schemas import ChatResponse, Source

logger = logging.getLogger(__name__)

class RAGEngine:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.vector_store = VectorStore()

    async def get_answer(self, session_id: str, query: str) -> ChatResponse:
        try:
            # 1. Generate Embedding for the query
            embed_res = await self.client.embeddings.create(
                input=query,
                model="text-embedding-3-small"
            )
            query_vector = embed_res.data[0].embedding

            # 2. Retrieve relevant chunks from Pinecone
            search_results = self.vector_store.query_vectors(
                vector=query_vector,
                session_id=session_id,
                top_k=5
            )

            context_parts = []
            sources = []
            
            matches = search_results.get("matches", [])
            if not matches:
                return ChatResponse(
                    answer="I couldn't find any relevant information in the uploaded documents to answer your question.",
                    sources=[]
                )

            for match in matches:
                metadata = match.get("metadata", {})
                text = metadata.get("text", "")
                doc_name = metadata.get("file_name", "Unknown Document")
                page_num = metadata.get("page_number", "N/A")
                
                context_parts.append(f"Source: {doc_name} (Page {page_num})\nContent: {text}")
                sources.append(Source(
                    document=doc_name,
                    page=page_num,
                    snippet=text[:200] + "..." if len(text) > 200 else text
                ))

            context_str = "\n\n---\n\n".join(context_parts)

            # 3. Construct Prompt and Generate Answer
            system_prompt = (
                "You are a professional assistant. Answer the user's question using ONLY the provided context. "
                "If the answer is not in the context, state that the answer was not found. "
                "Include citations in the format [Document Name - Page X]."
            )
            user_prompt = f"Context:\n{context_str}\n\nQuestion: {query}"

            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0
            )

            answer = response.choices[0].message.content

            return ChatResponse(answer=answer, sources=sources)

        except Exception as e:
            logger.error(f"Error in RAG Engine: {str(e)}")
            raise e
