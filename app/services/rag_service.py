from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from app.core.config import settings
from app.services.vector_service import VectorService
from app.models.schemas import ChatQueryResponse, Source

class RAGService:
    def __init__(self, vector_service: VectorService):
        self.vector_service = vector_service
        self.llm = ChatOpenAI(
            model=settings.MODEL_NAME,
            openai_api_key=settings.OPENAI_API_KEY,
            temperature=0
        )
        self.system_prompt = (
            "You are an assistant. Use ONLY the provided context to answer. "
            "If the answer is not in the context, say 'The answer was not found in the uploaded documents.' "
            "Include citations in your answer if possible, but the final 'sources' list will be handled separately."
            "\n\n"
            "Context:\n{context}"
        )

    async def answer_query(self, session_id: str, query: str) -> ChatQueryResponse:
        # 1. Retrieve relevant chunks
        results = self.vector_service.query(session_id, query)
        
        if not results['documents'] or not results['documents'][0]:
            return ChatQueryResponse(
                answer="The answer was not found in the uploaded documents.",
                sources=[]
            )

        context_text = "\n---\n".join(results['documents'][0])
        
        # 2. Build Prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{query}")
        ])
        
        chain = prompt | self.llm
        
        # 3. Generate Answer
        response = await chain.ainvoke({"context": context_text, "query": query})
        
        # 4. Extract Sources from metadata
        sources = []
        seen_sources = set()
        for meta in results['metadatas'][0]:
            source_key = (meta['doc_name'], meta['page_label'])
            if source_key not in seen_sources:
                sources.append(Source(document_name=meta['doc_name'], page=meta['page_label']))
                seen_sources.add(source_key)

        return ChatQueryResponse(
            answer=response.content,
            sources=sources
        )
