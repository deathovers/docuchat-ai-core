import openai
from typing import List, Dict
from api.core.config import settings
from api.services.vector_store import VectorStore
from api.models.schemas import Source

class RAGEngine:
    def __init__(self):
        self.vector_store = VectorStore()
        self.client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)

    async def get_answer(self, query: str, session_id: str) -> Dict:
        # Retrieve relevant chunks
        results = self.vector_store.query(query, session_id, top_k=5)
        
        if not results:
            return {
                "answer": "I couldn't find any relevant information in the uploaded documents.",
                "sources": []
            }

        # Construct context
        context_text = ""
        sources = []
        seen_sources = set()

        for res in results:
            context_text += f"\n---\nSource: {res.metadata['file_name']} (Page {res.metadata['page_number']})\nContent: {res.page_content}\n"
            
            source_key = f"{res.metadata['file_name']}-{res.metadata['page_number']}"
            if source_key not in seen_sources:
                sources.append(Source(
                    document=res.metadata['file_name'],
                    page=res.metadata['page_number'],
                    snippet=res.page_content[:200] + "..."
                ))
                seen_sources.add(source_key)

        # Generate response
        system_prompt = (
            "You are a professional assistant. Answer the user's question using ONLY the provided context. "
            "If the answer is not in the context, state that the answer was not found. "
            "Include citations in the format [Document Name - Page X] within your response."
        )
        
        user_prompt = f"Context:\n{context_text}\n\nQuestion: {query}"

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0
        )

        answer = response.choices[0].message.content

        return {
            "answer": answer,
            "sources": sources
        }
