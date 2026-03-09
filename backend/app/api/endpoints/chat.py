from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
from app.services.llm_service import llm_service
import json

router = APIRouter()

class ChatRequest(BaseModel):
    query: str
    session_id: str
    history: Optional[List[Dict[str, str]]] = []

@router.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Streaming chat endpoint that returns SSE.
    """
    async def event_generator():
        async for chunk in llm_service.get_chat_response(
            request.query, request.session_id, request.history
        ):
            # Check if this is the metadata chunk
            if "___DOCUCHAT_METADATA_SEPARATOR___" in chunk:
                parts = chunk.split("___DOCUCHAT_METADATA_SEPARATOR___")
                # Yield the text part if any
                if parts[0]:
                    yield f"data: {json.dumps({'type': 'text', 'content': parts[0]})}\n\n"
                # Yield the sources part
                if len(parts) > 1:
                    sources = json.loads(parts[1])
                    yield f"data: {json.dumps({'type': 'sources', 'content': sources})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'text', 'content': chunk})}\n\n"
        
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
