import json
import logging
from typing import List, Dict, Any, AsyncGenerator, Set
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from app.core.config import settings

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        try:
            self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            self.llm = ChatOpenAI(model="gpt-4o", temperature=0, streaming=True)
            self.vector_store = PineconeVectorStore(
                index_name=settings.PINECONE_INDEX,
                embedding=self.embeddings,
                pinecone_api_key=settings.PINECONE_API_KEY
            )
        except Exception as e:
            logger.error(f"Failed to initialize LLMService: {str(e)}")
            raise

    async def get_chat_response(
        self, query: str, session_id: str, history: List[Dict[str, str]]
    ) -> AsyncGenerator[str, None]:
        """
        Generates a streaming RAG response with conversation history and citations.
        """
        try:
            # 1. Retrieve relevant chunks with strict session filtering
            # Stability: Wrap vector retrieval in try-except
            try:
                retriever = self.vector_store.as_retriever(
                    search_kwargs={"filter": {"session_id": session_id}, "k": 5}
                )
                docs = await retriever.ainvoke(query)
            except Exception as e:
                logger.error(f"Vector retrieval error: {str(e)}")
                yield "I'm sorry, I encountered an error while searching your documents. Please try again later."
                return

            # 2. Prepare context and deduplicate sources
            # UX: Sources list deduplication
            context_parts = []
            unique_sources: Set[tuple] = set()
            sources_metadata = []

            for d in docs:
                file_name = d.metadata.get('file_name', 'Unknown')
                page_num = d.metadata.get('page_number', 'N/A')
                
                context_parts.append(f"Source: {file_name} (Page {page_num})\nContent: {d.page_content}")
                
                source_tuple = (file_name, page_num)
                if source_tuple not in unique_sources:
                    unique_sources.add(source_tuple)
                    sources_metadata.append({"file_name": file_name, "page": page_num})

            context = "\n\n".join(context_parts)

            # 3. Construct the RAG prompt with History
            # CRITICAL: history parameter is now integrated
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant. Answer the question using ONLY the provided context. "
                           "If the answer is not in the context, say you don't know. "
                           "Always cite your sources using [File Name, Page X] format.\n\n"
                           f"Context:\n{context}"),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{query}")
            ])

            # Convert history dicts to LangChain message objects
            formatted_history = []
            for msg in history:
                if msg["role"] == "user":
                    formatted_history.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    formatted_history.append(AIMessage(content=msg["content"]))

            # 4. Stream the response
            # Stability: Wrap LLM streaming in try-except
            try:
                chain = prompt | self.llm
                async for chunk in chain.astream({"history": formatted_history, "query": query}):
                    yield chunk.content
            except Exception as e:
                logger.error(f"LLM streaming error: {str(e)}")
                yield "\n[Error generating response. Please check your connection.]"
                return

            # 5. Append sources metadata at the end
            # RISK: Using a more unique delimiter to avoid collision
            delimiter = "___DOCUCHAT_METADATA_SEPARATOR___"
            yield f"{delimiter}{json.dumps(sources_metadata)}"

        except Exception as e:
            logger.error(f"Unexpected error in get_chat_response: {str(e)}")
            yield "An unexpected error occurred. Please try again."

llm_service = LLMService()
