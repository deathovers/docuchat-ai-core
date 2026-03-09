import logging
import json
import tiktoken
from typing import List, Dict, Any, Tuple
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, AIMessage, SystemMessage, BaseMessage

# Configure logging for observability
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self, api_key: str, model_name: str = "gpt-4o"):
        if not api_key:
            raise ValueError("OpenAI API Key is required.")
        
        self.llm = ChatOpenAI(
            api_key=api_key, 
            model=model_name, 
            streaming=True,
            temperature=0
        )
        
        try:
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        except KeyError:
            logger.warning(f"Model {model_name} not found in tiktoken, falling back to cl100k_base.")
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            
        self.max_tokens = 128000  # Context window for GPT-4o
        self.reserve_tokens = 4000  # Reserve for response and system prompt overhead

    def _validate_history(self, history: List[Dict[str, str]]):
        """Validates the structure of the chat history to prevent crashes."""
        if not isinstance(history, list):
            raise ValueError("History must be a list of message objects.")
        for i, msg in enumerate(history):
            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                raise ValueError(f"Malformed history item at index {i}. Must be dict with 'role' and 'content'.")
            if msg["role"] not in ["user", "assistant"]:
                raise ValueError(f"Invalid role '{msg['role']}' at index {i}.")

    def _count_tokens(self, text: str) -> int:
        """Counts the number of tokens in a string."""
        return len(self.tokenizer.encode(text))

    def _truncate_history(self, history: List[BaseMessage], limit: int) -> List[BaseMessage]:
        """
        Implements a sliding window for history based on token budget.
        Keeps the most recent messages that fit within the limit.
        """
        truncated = []
        current_tokens = 0
        # Iterate backwards to keep most recent messages
        for msg in reversed(history):
            msg_tokens = self._count_tokens(msg.content)
            if current_tokens + msg_tokens > limit:
                logger.info(f"Token limit reached ({limit}). Truncating older history messages.")
                break
            truncated.insert(0, msg)
            current_tokens += msg_tokens
        return truncated

    async def get_chat_response(
        self, 
        query: str, 
        history: List[Dict[str, str]], 
        context_chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Orchestrates the RAG chat response generation with token management and error handling.
        """
        try:
            # 1. Input Validation
            self._validate_history(history)
            if not query:
                raise ValueError("Query cannot be empty.")

            # 2. Convert history to LangChain messages
            chat_history: List[BaseMessage] = []
            for msg in history:
                if msg["role"] == "user":
                    chat_history.append(HumanMessage(content=msg["content"]))
                else:
                    chat_history.append(AIMessage(content=msg["content"]))

            # 3. Deduplicate Sources and Prepare Context
            unique_sources = set()
            context_parts = []
            for chunk in context_chunks:
                metadata = chunk.get("metadata", {})
                name = metadata.get("document_name", "Unknown Document")
                page = metadata.get("page_number", "N/A")
                
                unique_sources.add((name, page))
                context_parts.append(f"Source: {name} (Page {page})\nContent: {chunk.get('text', '')}")

            context_text = "\n\n---\n\n".join(context_parts)

            # 4. Token Management
            query_tokens = self._count_tokens(query)
            context_tokens = self._count_tokens(context_text)
            
            # Calculate available budget for history
            available_for_history = self.max_tokens - self.reserve_tokens - context_tokens - query_tokens
            
            # Ensure a minimum floor for history or handle extreme context size
            if available_for_history < 1000:
                logger.warning("Context is extremely large. Truncating context to preserve history budget.")
                context_text = context_text[:20000] # Naive truncation for safety
                context_tokens = self._count_tokens(context_text)
                available_for_history = self.max_tokens - self.reserve_tokens - context_tokens - query_tokens

            chat_history = self._truncate_history(chat_history, available_for_history)

            # 5. Prompt Construction
            prompt = ChatPromptTemplate.from_messages([
                ("system", (
                    "You are DocuChat AI, a professional assistant. "
                    "Use the provided context to answer the user's question accurately. "
                    "If the answer is not in the context, say you don't know. "
                    "Always cite sources in the text like [Document Name, Page X]. "
                    "After your response, append the delimiter '__DOCUCHAT_SOURCES_JSON__' "
                    "followed by a JSON list of the sources you used."
                )),
                MessagesPlaceholder(variable_name="history"),
                ("human", f"Context:\n{context_text}\n\nQuestion: {query}")
            ])

            # 6. Execution
            chain = prompt | self.llm
            full_response = ""
            
            try:
                async for chunk in chain.astream({"history": chat_history}):
                    full_response += chunk.content
            except Exception as e:
                logger.error(f"LLM Streaming Error: {str(e)}", exc_info=True)
                raise RuntimeError("Failed to stream response from OpenAI.")

            # 7. Post-processing and Source Extraction
            parts = full_response.split("__DOCUCHAT_SOURCES_JSON__")
            answer = parts[0].strip()
            
            # We return the deduplicated sources gathered from the context chunks
            sources_list = [
                {"document_name": name, "page_number": page} 
                for name, page in sorted(list(unique_sources))
            ]
            
            return {
                "answer": answer,
                "sources": sources_list
            }

        except ValueError as ve:
            logger.error(f"Validation Error: {str(ve)}")
            return {"answer": f"Input error: {str(ve)}", "sources": []}
        except Exception as e:
            logger.error(f"Critical Error in LLMService: {str(e)}", exc_info=True)
            return {
                "answer": "I'm sorry, I encountered an internal error while processing your request.",
                "sources": []
            }
