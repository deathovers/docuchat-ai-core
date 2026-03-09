import logging
import json
import tiktoken
from typing import List, Dict, Any, Tuple, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, AIMessage, SystemMessage, BaseMessage

# Standard logger - basicConfig removed to avoid overriding global settings
# This allows the parent application (FastAPI) to control logging configuration.
logger = logging.getLogger(__name__)

class LLMService:
    """
    Orchestrates the LLM interaction for DocuChat AI, handling token management,
    sliding window history, and accurate source attribution.
    """
    def __init__(self, api_key: str, model_name: str = "gpt-4o"):
        if not api_key:
            raise ValueError("OpenAI API Key is required.")
        
        self.model_name = model_name
        self.llm = ChatOpenAI(
            api_key=api_key, 
            model=model_name, 
            streaming=True,
            temperature=0
        )
        
        try:
            # Initialize tokenizer for precise token counting
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        except KeyError:
            logger.warning(f"Model {model_name} not found in tiktoken, falling back to cl100k_base.")
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            
        self.max_tokens = 128000  # Context window for GPT-4o
        self.reserve_tokens = 4000  # Reserve for response and system prompt overhead
        self.delimiter = "__DOCUCHAT_SOURCES_JSON__"

    def _count_tokens(self, text: str) -> int:
        """Counts the number of tokens in a string using tiktoken."""
        if not text:
            return 0
        return len(self.tokenizer.encode(text))

    def _truncate_tokens(self, text: str, max_tokens: int) -> str:
        """
        Truncates a string to a specific token count safely.
        Prevents UTF-8 corruption by slicing on token boundaries.
        """
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self.tokenizer.decode(tokens[:max_tokens])

    def _truncate_history(self, history: List[BaseMessage], limit: int) -> List[BaseMessage]:
        """
        Implements a sliding window for history based on token budget.
        Keeps the most recent messages that fit within the limit.
        """
        truncated = []
        current_tokens = 0
        # Iterate backwards to keep the most recent context
        for msg in reversed(history):
            msg_tokens = self._count_tokens(msg.content)
            if current_tokens + msg_tokens > limit:
                logger.debug(f"History token limit reached ({limit}). Truncating older messages.")
                break
            truncated.insert(0, msg)
            current_tokens += msg_tokens
        return truncated

    def _validate_history(self, history: List[Dict[str, str]]):
        """Validates the structure of the chat history to prevent crashes."""
        if not isinstance(history, list):
            raise ValueError("History must be a list of message objects.")
        for i, msg in enumerate(history):
            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                raise ValueError(f"Malformed history item at index {i}. Must be dict with 'role' and 'content'.")
            if msg["role"] not in ["user", "assistant"]:
                raise ValueError(f"Invalid role '{msg['role']}' at index {i}.")

    async def get_chat_response(
        self, 
        query: str, 
        history: List[Dict[str, str]], 
        context_chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Orchestrates the RAG chat response generation with token management and source attribution.
        
        Args:
            query: The user's question.
            history: List of previous messages in the session.
            context_chunks: List of retrieved document chunks with metadata.
            
        Returns:
            Dict containing the 'answer' and a list of 'sources' used.
        """
        try:
            # 1. Input Validation
            self._validate_history(history)
            if not query:
                raise ValueError("Query cannot be empty.")

            # 2. Prepare Context and Deduplicate Sources for potential fallback
            context_parts = []
            available_sources = []
            for chunk in context_chunks:
                meta = chunk.get("metadata", {})
                source_info = {
                    "document_name": meta.get("document_name", "Unknown Document"),
                    "page_number": meta.get("page_number", "N/A")
                }
                available_sources.append(source_info)
                context_parts.append(
                    f"Source: {source_info['document_name']} (Page {source_info['page_number']})\n"
                    f"Content: {chunk.get('text', '')}"
                )

            full_context_text = "\n\n---\n\n".join(context_parts)

            # 3. Token Budgeting & Truncation
            query_tokens = self._count_tokens(query)
            
            # Calculate remaining space after query and reserves
            remaining_budget = self.max_tokens - self.reserve_tokens - query_tokens
            
            # Prioritize context (70%) over history (30%)
            context_budget = int(remaining_budget * 0.7)
            history_budget = remaining_budget - context_budget

            # Truncate context using tokens (Fixes CRITICAL bug: Naive slicing)
            truncated_context = self._truncate_tokens(full_context_text, context_budget)

            # 4. Process History with Sliding Window
            chat_history: List[BaseMessage] = []
            for msg in history:
                if msg["role"] == "user":
                    chat_history.append(HumanMessage(content=msg["content"]))
                else:
                    chat_history.append(AIMessage(content=msg["content"]))
            
            truncated_history = self._truncate_history(chat_history, history_budget)

            # 5. Prompt Construction
            prompt = ChatPromptTemplate.from_messages([
                ("system", (
                    "You are DocuChat AI, a professional assistant. "
                    "Use the provided context to answer the user's question accurately. "
                    "If the answer is not in the context, state that you do not have enough information. "
                    "Always cite sources in the text like [Document Name, Page X]. "
                    f"After your response, append the delimiter '{self.delimiter}' "
                    "followed by a JSON list of the sources you actually used to construct the answer. "
                    "Example: [{\"document_name\": \"Doc.pdf\", \"page_number\": 1}]"
                )),
                MessagesPlaceholder(variable_name="history"),
                ("human", f"Context:\n{truncated_context}\n\nQuestion: {query}")
            ])

            # 6. Execution (Streaming)
            chain = prompt | self.llm
            full_response = ""
            
            try:
                async for chunk in chain.astream({"history": truncated_history}):
                    full_response += chunk.content
            except Exception as e:
                logger.error(f"LLM Streaming Error: {str(e)}", exc_info=True)
                raise RuntimeError("Failed to stream response from OpenAI.")

            # 7. Post-processing and Source Extraction (Fixes CRITICAL bug: Source Over-inclusion)
            parts = full_response.split(self.delimiter)
            answer = parts[0].strip()
            final_sources = []

            if len(parts) > 1:
                try:
                    # Attempt to parse LLM-generated JSON for precise attribution
                    json_str = parts[1].strip()
                    final_sources = json.loads(json_str)
                except json.JSONDecodeError:
                    logger.warning("LLM failed to provide valid JSON sources. Falling back to context chunks.")
                    # Fallback: Deduplicate and return all provided context sources if JSON fails
                    seen = set()
                    for s in available_sources:
                        identifier = f"{s['document_name']}-{s['page_number']}"
                        if identifier not in seen:
                            final_sources.append(s)
                            seen.add(identifier)
            else:
                logger.warning("LLM response missing source delimiter. Returning empty source list.")

            return {
                "answer": answer,
                "sources": final_sources
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
